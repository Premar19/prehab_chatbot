"""
RAG Retrieval Evaluation Script
--------------------------------
Runs the gold-standard test set through the FAISS retrieval
pipeline and computes standard information retrieval metrics:

    - Recall@k       for k in {1, 3, 5}
    - Mean Reciprocal Rank (MRR)
    - Precision@5
    - Out-of-scope handling accuracy

Results are broken down by question category and query type,
printed as tables, and saved to a JSON report file for use in
the dissertation Testing chapter.

Run from the project root:
    python evaluate_retrieval.py

References:
    Manning, C.D., Raghavan, P., & Schütze, H. (2008).
    Introduction to Information Retrieval. Cambridge Univ. Press.
"""

import json
import numpy as np
import faiss
from collections import defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────
FAISS_INDEX_PATH = "data/index/faiss_index.bin"
METADATA_PATH = "data/index/faiss_metadata.json"
TEST_SET_PATH = "data/evaluation/gold-standard-test-set.json"
REPORT_PATH = "data/evaluation/evaluation_report.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
MIN_SCORE = 0.45  # Same threshold as the server


# ── Metric helpers ─────────────────────────────────────────────

def recall_at_k(retrieved_ids, expected_ids, k):
    """
    Recall@k — was at least one expected chunk in the top-k retrieved?
    Returns 1.0 if yes, 0.0 if no.

    For questions with multiple expected chunks, this measures
    'any-match' recall. A stricter variant is computed separately.
    """
    if not expected_ids:
        return None  # Out-of-scope: not applicable here
    top_k = retrieved_ids[:k]
    return 1.0 if any(e in top_k for e in expected_ids) else 0.0


def mean_recall_at_k(retrieved_ids, expected_ids, k):
    """
    Fractional recall@k — what fraction of expected chunks appear
    in the top-k? Useful when expected_ids has multiple items.
    """
    if not expected_ids:
        return None
    top_k = set(retrieved_ids[:k])
    matches = sum(1 for e in expected_ids if e in top_k)
    return matches / len(expected_ids)


def reciprocal_rank(retrieved_ids, expected_ids):
    """
    Reciprocal rank — 1 / rank of the first expected chunk in the
    retrieved list. Returns 0.0 if no expected chunk was found.
    """
    if not expected_ids:
        return None
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_ids, expected_ids, k):
    """
    Precision@k — what fraction of the top-k retrieved chunks
    are actually in the expected set?
    """
    if not expected_ids:
        return None
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    matches = sum(1 for r in top_k if r in expected_ids)
    return matches / len(top_k)


# ── Retrieval function (mirrors server.py logic) ───────────────

def search_faiss(query, history, model, index, chunks, top_k=TOP_K):
    """
    Search the FAISS index for a query, optionally expanding the
    query with recent user messages from the history. The current
    query is repeated (term repetition weighting) so its terms
    dominate the embedding over the historical context.
    Applies the MIN_SCORE threshold identically to server.py.
    Returns a list of (chunk_id, score) tuples.
    """
    search_query = query
    if history:
        previous_user_msgs = [m["text"] for m in history if m["role"] == "user"]
        if previous_user_msgs:
            recent_context = " ".join(previous_user_msgs[-3:])
            # Repeat the current query to give it more weight
            search_query = f"{recent_context} {query} {query}"

    query_vector = model.encode(
        [search_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(query_vector, k=top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if float(score) >= MIN_SCORE:
            results.append((chunks[idx]["chunk_id"], float(score)))

    return results


# ── Reporting helpers ──────────────────────────────────────────

def format_table(headers, rows):
    """Print a simple ASCII table."""
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2
                  for i in range(len(headers))]

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    header_row = "|" + "|".join(
        f" {headers[i]:<{col_widths[i]-1}}" for i in range(len(headers))
    ) + "|"

    lines = [sep, header_row, sep]
    for row in rows:
        row_str = "|" + "|".join(
            f" {str(row[i]):<{col_widths[i]-1}}" for i in range(len(row))
        ) + "|"
        lines.append(row_str)
    lines.append(sep)
    return "\n".join(lines)


# ── Main evaluation logic ──────────────────────────────────────

def main():
    print("=" * 65)
    print("CKD Chatbot RAG Retrieval Evaluation")
    print("=" * 65)

    # Load resources
    print("\nLoading FAISS index and metadata...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    chunks = metadata["chunks"]
    print(f"  Loaded {index.ntotal} vectors ({len(chunks)} chunks)")

    print(f"\nLoading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\nLoading gold-standard test set...")
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    questions = test_set["questions"]
    print(f"  Loaded {len(questions)} questions")
    print()

    # ── Run each question through the pipeline ────────────────
    results = []
    for q in questions:
        retrieved = search_faiss(
            query=q["question"],
            history=q.get("history", []),
            model=model,
            index=index,
            chunks=chunks,
        )
        retrieved_ids = [r[0] for r in retrieved]
        retrieved_scores = [r[1] for r in retrieved]

        result = {
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "query_type": q["query_type"],
            "expected_chunk_ids": q["expected_chunk_ids"],
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
        }

        # Compute metrics
        if q["query_type"] == "out_of_scope":
            # Success = retrieval returned NOTHING
            result["out_of_scope_pass"] = len(retrieved_ids) == 0
        else:
            result["recall@1"] = recall_at_k(retrieved_ids, q["expected_chunk_ids"], 1)
            result["recall@3"] = recall_at_k(retrieved_ids, q["expected_chunk_ids"], 3)
            result["recall@5"] = recall_at_k(retrieved_ids, q["expected_chunk_ids"], 5)
            result["mean_recall@5"] = mean_recall_at_k(
                retrieved_ids, q["expected_chunk_ids"], 5
            )
            result["mrr"] = reciprocal_rank(retrieved_ids, q["expected_chunk_ids"])
            result["precision@5"] = precision_at_k(
                retrieved_ids, q["expected_chunk_ids"], 5
            )

        results.append(result)

    # ── Aggregate metrics ──────────────────────────────────────
    in_scope = [r for r in results if r["query_type"] != "out_of_scope"]
    out_scope = [r for r in results if r["query_type"] == "out_of_scope"]

    overall = {
        "total_questions": len(results),
        "in_scope_questions": len(in_scope),
        "out_of_scope_questions": len(out_scope),
        "avg_recall@1": round(np.mean([r["recall@1"] for r in in_scope]), 3),
        "avg_recall@3": round(np.mean([r["recall@3"] for r in in_scope]), 3),
        "avg_recall@5": round(np.mean([r["recall@5"] for r in in_scope]), 3),
        "avg_mean_recall@5": round(np.mean([r["mean_recall@5"] for r in in_scope]), 3),
        "avg_mrr": round(np.mean([r["mrr"] for r in in_scope]), 3),
        "avg_precision@5": round(np.mean([r["precision@5"] for r in in_scope]), 3),
        "out_of_scope_accuracy": round(
            sum(1 for r in out_scope if r["out_of_scope_pass"]) / len(out_scope), 3
        ) if out_scope else None,
    }

    # ── Breakdown by category ──────────────────────────────────
    by_category = defaultdict(list)
    for r in in_scope:
        by_category[r["category"]].append(r)

    category_metrics = {}
    for cat, items in by_category.items():
        category_metrics[cat] = {
            "n": len(items),
            "recall@1": round(np.mean([r["recall@1"] for r in items]), 3),
            "recall@3": round(np.mean([r["recall@3"] for r in items]), 3),
            "recall@5": round(np.mean([r["recall@5"] for r in items]), 3),
            "mrr": round(np.mean([r["mrr"] for r in items]), 3),
        }

    # ── Breakdown by query type ────────────────────────────────
    by_type = defaultdict(list)
    for r in in_scope:
        by_type[r["query_type"]].append(r)

    query_type_metrics = {}
    for qt, items in by_type.items():
        query_type_metrics[qt] = {
            "n": len(items),
            "recall@1": round(np.mean([r["recall@1"] for r in items]), 3),
            "recall@3": round(np.mean([r["recall@3"] for r in items]), 3),
            "recall@5": round(np.mean([r["recall@5"] for r in items]), 3),
            "mrr": round(np.mean([r["mrr"] for r in items]), 3),
        }

    # ── Print results ──────────────────────────────────────────
    print("=" * 65)
    print("OVERALL RESULTS")
    print("=" * 65)
    print(f"  Total questions:        {overall['total_questions']}")
    print(f"  In-scope questions:     {overall['in_scope_questions']}")
    print(f"  Out-of-scope questions: {overall['out_of_scope_questions']}")
    print()
    print(f"  Recall@1:              {overall['avg_recall@1']:.3f}  ({overall['avg_recall@1']*100:.1f}%)")
    print(f"  Recall@3:              {overall['avg_recall@3']:.3f}  ({overall['avg_recall@3']*100:.1f}%)")
    print(f"  Recall@5:              {overall['avg_recall@5']:.3f}  ({overall['avg_recall@5']*100:.1f}%)")
    print(f"  Mean Recall@5:         {overall['avg_mean_recall@5']:.3f}")
    print(f"  MRR:                   {overall['avg_mrr']:.3f}")
    print(f"  Precision@5:           {overall['avg_precision@5']:.3f}")
    print(f"  Out-of-scope accuracy: {overall['out_of_scope_accuracy']:.3f}  ({overall['out_of_scope_accuracy']*100:.1f}%)")

    print("\n" + "=" * 65)
    print("BREAKDOWN BY CATEGORY")
    print("=" * 65)
    rows = [
        [cat, m["n"], f"{m['recall@1']:.2f}", f"{m['recall@3']:.2f}",
         f"{m['recall@5']:.2f}", f"{m['mrr']:.3f}"]
        for cat, m in sorted(category_metrics.items())
    ]
    print(format_table(
        ["Category", "N", "R@1", "R@3", "R@5", "MRR"],
        rows,
    ))

    print("\n" + "=" * 65)
    print("BREAKDOWN BY QUERY TYPE")
    print("=" * 65)
    rows = [
        [qt, m["n"], f"{m['recall@1']:.2f}", f"{m['recall@3']:.2f}",
         f"{m['recall@5']:.2f}", f"{m['mrr']:.3f}"]
        for qt, m in sorted(query_type_metrics.items())
    ]
    print(format_table(
        ["Query Type", "N", "R@1", "R@3", "R@5", "MRR"],
        rows,
    ))

    # ── Show failures ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FAILED RETRIEVALS (Recall@5 == 0)")
    print("=" * 65)
    failed = [r for r in in_scope if r.get("recall@5") == 0.0]
    if failed:
        for r in failed:
            print(f"\n  {r['id']} [{r['query_type']}] \"{r['question']}\"")
            print(f"    Expected: {r['expected_chunk_ids']}")
            print(f"    Got:      {r['retrieved_chunk_ids'][:3]}...")
    else:
        print("  None! All in-scope questions had at least one expected chunk in top 5.")

    # Out-of-scope failures (retrieval returned something for an off-topic question)
    os_failed = [r for r in out_scope if not r["out_of_scope_pass"]]
    if os_failed:
        print("\n" + "=" * 65)
        print("OUT-OF-SCOPE FAILURES (threshold didn't filter them)")
        print("=" * 65)
        for r in os_failed:
            print(f"\n  {r['id']} \"{r['question']}\"")
            print(f"    Unexpectedly retrieved: {r['retrieved_chunk_ids'][:3]}")
            print(f"    Scores: {[f'{s:.3f}' for s in r['retrieved_scores'][:3]]}")

    # ── Save full report ───────────────────────────────────────
    report = {
        "metadata": {
            "evaluated_at": datetime.now().isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K,
            "min_score_threshold": MIN_SCORE,
            "test_set_version": test_set["metadata"].get("version", "1.0"),
        },
        "overall": overall,
        "by_category": category_metrics,
        "by_query_type": query_type_metrics,
        "per_question": results,
    }

    import os
    os.makedirs("data/evaluation", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n\n✓ Full report saved to: {REPORT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()