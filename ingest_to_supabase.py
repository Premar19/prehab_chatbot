"""
Ingest NHS CKD chunks into Supabase (pgvector) — one-off loader.
-----------------------------------------------------------------
Reads nhs_ckd_chunks.json, embeds each chunk's content with the SAME model
your app uses (all-MiniLM-L6-v2, normalized — so cosine scores match your old
FAISS scores), and upserts rows into the existing `rag_chunks` table.

Idempotent: re-running updates existing rows by doc_id rather than duplicating
(requires the unique index on doc_id from step 1a).

Setup:
    pip install supabase sentence-transformers python-dotenv
    Credentials are read from .env (never hardcode them here):
        SUPABASE_URL=...
        SUPABASE_SERVICE_KEY=...   # backend only, keep secret

Run:
    python ingest_to_supabase.py nhs_ckd_chunks.json
"""

import os
import sys
import json
import re
import hashlib

from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TABLE = "rag_chunks"
BATCH = 25  # rows per upsert request

# chunk_id looks like "nhs-kidney-disease-...-overview-001" -> trailing number is the index
CHUNK_INDEX_RE = re.compile(r"-(\d+)$")


def to_vector_literal(vec) -> str:
    """pgvector text format '[v1,v2,...]'. Sent as a string so PostgREST casts
    it to vector reliably (a bare JSON array can be misread as a Postgres array)."""
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


def build_rows(data, embeddings):
    source = data.get("metadata", {}).get("source", "NHS UK")
    rows = []
    for i, (c, emb) in enumerate(zip(data["chunks"], embeddings)):
        m = CHUNK_INDEX_RE.search(c["chunk_id"])
        rows.append({
            "doc_id": c["chunk_id"],
            "source": source,
            "source_url": c["source_url"],
            "title": c["page_title"],
            "section": c["section_title"],
            "chunk_index": int(m.group(1)) if m else i,
            "content": c["content"],
            "content_hash": hashlib.sha256(c["content"].encode("utf-8")).hexdigest(),
            "embedding": to_vector_literal(emb),
        })
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "nhs_ckd_chunks.json"

    try:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
    except KeyError as e:
        sys.exit(f"Missing environment variable: {e}. See the setup notes at the top of this file.")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    print(f"Loaded {len(chunks)} chunks from {path}")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding content (normalized, to match FAISS cosine scores)...")
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=True,
    ).tolist()

    rows = build_rows(data, embeddings)

    print(f"Upserting {len(rows)} rows into '{TABLE}' in batches of {BATCH}...")
    sb = create_client(url, key)
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        sb.table(TABLE).upsert(batch, on_conflict="doc_id").execute()
        print(f"  upserted {min(i + BATCH, len(rows))}/{len(rows)}")

    res = sb.table(TABLE).select("id", count="exact").execute()
    print(f"Done. '{TABLE}' now reports {res.count} rows.")


if __name__ == "__main__":
    main()