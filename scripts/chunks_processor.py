"""
NHS CKD Chunk Processor
-----------------------
Takes the raw scraped nhs_ckd_chunks.json and:
1. Removes junk chunks (link-only, navigation text, too short)
2. Splits oversized chunks into ~300-400 word sub-chunks with overlap
3. Outputs a clean file ready for FAISS indexing

Run: python process_chunks.py
Input:  nhs_ckd_chunks.json  (from scraper)
Output: nhs_ckd_chunks_processed.json (ready for FAISS)
"""

import json
import re
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────
MAX_CHUNK_WORDS = 400       # Split chunks larger than this
OVERLAP_WORDS = 50          # Word overlap between split chunks
MIN_CHUNK_WORDS = 25        # Discard chunks smaller than this

# Section titles that are just external links, not useful content
JUNK_SECTION_TITLES = [
    "want to know more?",
    "social care and support guide",
]

# Content patterns that indicate junk (navigation text, etc.)
JUNK_CONTENT_PATTERNS = [
    r"^what it is\s+why it is done",       # nav-only text
    r"^find out more about",                # link-only
    r"^video:",                             # video placeholder
]


def is_junk_chunk(chunk: dict) -> tuple:
    """
    Check if a chunk is junk. Returns (is_junk: bool, reason: str).
    """
    section = chunk["section_title"].lower().strip()
    content = chunk["content"].strip()
    word_count = chunk["word_count"]

    # Check section title
    for junk_title in JUNK_SECTION_TITLES:
        if section == junk_title:
            return True, f"Junk section title: '{section}'"

    # Check content patterns
    for pattern in JUNK_CONTENT_PATTERNS:
        if re.match(pattern, content.lower()):
            return True, f"Junk content pattern: '{content[:60]}...'"

    # Too short to be useful
    if word_count < MIN_CHUNK_WORDS:
        return True, f"Too short: {word_count} words"

    # Content is mostly just external resource names (heuristic: lots of colons)
    colon_count = content.count(":")
    words = content.split()
    if colon_count > 3 and word_count < 60:
        return True, f"Appears to be link list ({colon_count} colons in {word_count} words)"

    return False, ""


def split_chunk(chunk: dict, max_words: int, overlap: int) -> list:
    """
    Split an oversized chunk into smaller sub-chunks with word overlap.
    Tries to split at sentence boundaries for clean breaks.
    """
    words = chunk["content"].split()

    if len(words) <= max_words:
        return [chunk]

    # Split content into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', chunk["content"])

    sub_chunks = []
    current_sentences = []
    current_word_count = 0
    part_num = 1

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # If adding this sentence would exceed the limit, save current chunk
        if current_word_count + sentence_words > max_words and current_sentences:
            # Build the sub-chunk
            sub_content = " ".join(current_sentences)
            sub_chunks.append({
                "page_title": chunk["page_title"],
                "section_title": f"{chunk['section_title']} (part {part_num})",
                "content": sub_content,
                "source_url": chunk["source_url"],
                "chunk_id": f"{chunk['chunk_id']}-p{part_num}",
                "word_count": len(sub_content.split()),
            })
            part_num += 1

            # Keep last few sentences as overlap for context continuity
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_words = len(s.split())
                if overlap_count + s_words > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += s_words

            current_sentences = overlap_sentences
            current_word_count = overlap_count

        current_sentences.append(sentence)
        current_word_count += sentence_words

    # Don't forget the last chunk
    if current_sentences:
        sub_content = " ".join(current_sentences)
        # Only add if it has meaningful content beyond just the overlap
        if len(sub_content.split()) > overlap + 10:
            sub_chunks.append({
                "page_title": chunk["page_title"],
                "section_title": f"{chunk['section_title']} (part {part_num})",
                "content": sub_content,
                "source_url": chunk["source_url"],
                "chunk_id": f"{chunk['chunk_id']}-p{part_num}",
                "word_count": len(sub_content.split()),
            })
        elif sub_chunks:
            # Merge remainder into the last chunk if it's too small
            last = sub_chunks[-1]
            merged = last["content"] + " " + " ".join(
                s for s in current_sentences
                if s not in last["content"]
            )
            last["content"] = merged.strip()
            last["word_count"] = len(last["content"].split())

    return sub_chunks


def main():
    # Load raw chunks
    with open("nhs_ckd_chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_chunks = data["chunks"]
    print(f"Loaded {len(raw_chunks)} raw chunks ({data['metadata']['total_words']} words)\n")

    # ── Step 1: Filter junk ──
    print("=" * 60)
    print("STEP 1: Filtering junk chunks")
    print("=" * 60)

    clean_chunks = []
    removed = []

    for chunk in raw_chunks:
        is_junk, reason = is_junk_chunk(chunk)
        if is_junk:
            removed.append((chunk["chunk_id"], reason))
            print(f"  ✗ Removed: {chunk['chunk_id']}")
            print(f"    Reason:  {reason}")
        else:
            clean_chunks.append(chunk)

    print(f"\n  Kept: {len(clean_chunks)} | Removed: {len(removed)}\n")

    # ── Step 2: Split oversized chunks ──
    print("=" * 60)
    print(f"STEP 2: Splitting chunks > {MAX_CHUNK_WORDS} words")
    print("=" * 60)

    final_chunks = []
    splits_done = 0

    for chunk in clean_chunks:
        if chunk["word_count"] > MAX_CHUNK_WORDS:
            sub_chunks = split_chunk(chunk, MAX_CHUNK_WORDS, OVERLAP_WORDS)
            print(f"  ✂ Split: {chunk['chunk_id']} ({chunk['word_count']} words) → {len(sub_chunks)} parts")
            for sc in sub_chunks:
                print(f"      → {sc['chunk_id']} ({sc['word_count']} words)")
            final_chunks.extend(sub_chunks)
            splits_done += 1
        else:
            final_chunks.append(chunk)

    print(f"\n  Chunks split: {splits_done}")
    print(f"  Final chunk count: {len(final_chunks)}\n")

    # ── Step 3: Summary statistics ──
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    word_counts = [c["word_count"] for c in final_chunks]
    total_words = sum(word_counts)

    print(f"  Total chunks:    {len(final_chunks)}")
    print(f"  Total words:     {total_words}")
    print(f"  Avg chunk size:  {total_words // len(final_chunks)} words")
    print(f"  Smallest chunk:  {min(word_counts)} words")
    print(f"  Largest chunk:   {max(word_counts)} words")

    # Flag any remaining large chunks
    large = [c for c in final_chunks if c["word_count"] > MAX_CHUNK_WORDS + 50]
    if large:
        print(f"\n  ⚠ {len(large)} chunks still over {MAX_CHUNK_WORDS + 50} words:")
        for c in large:
            print(f"    {c['chunk_id']}: {c['word_count']} words")

    # ── Step 4: Save ──
    output = {
        "metadata": {
            "processed_at": datetime.now().isoformat(),
            "source": "NHS UK",
            "topic": "Chronic Kidney Disease (CKD)",
            "total_chunks": len(final_chunks),
            "total_words": total_words,
            "avg_chunk_words": total_words // len(final_chunks),
            "processing_config": {
                "max_chunk_words": MAX_CHUNK_WORDS,
                "overlap_words": OVERLAP_WORDS,
                "min_chunk_words": MIN_CHUNK_WORDS,
            },
            "raw_chunks": len(raw_chunks),
            "chunks_removed": len(removed),
        },
        "chunks": final_chunks,
    }

    output_path = "nhs_ckd_chunks_processed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Saved to {output_path}")
    print()

    # ── Show preview ──
    print("=" * 60)
    print("PREVIEW (first 5 chunks)")
    print("=" * 60)
    for chunk in final_chunks[:5]:
        print(f"\n  ID:      {chunk['chunk_id']}")
        print(f"  Section: {chunk['section_title']}")
        print(f"  Words:   {chunk['word_count']}")
        print(f"  Content: {chunk['content'][:120]}...")


if __name__ == "__main__":
    main()