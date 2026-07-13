"""
embed_pipeline.py — RegulAI RAG Indexing Pipeline
==================================================
Reads documents from the 'notifications' MongoDB collection (already
populated by mongo_pipeline.py, including full_text), splits each
document's full_text into overlapping chunks, embeds each chunk with
a local Sentence Transformers model, and upserts the chunks (with
their embeddings) into a separate 'chunks' collection that Atlas
Vector Search queries against.

Run this AFTER mongo_pipeline.py, and re-run it any time you scrape
new documents — it's idempotent (safe to re-run, upserts by chunk_id).

Usage:
    python embed_pipeline.py

Requirements:
    pip install pymongo sentence-transformers python-dotenv
"""
from __future__ import annotations

import os
from datetime import datetime
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MONGO_URI         = os.getenv("MONGO_URI", "YOUR_MONGODB_ATLAS_URI_HERE")
DB_NAME           = "regulai"
SOURCE_COLLECTION = "notifications"
CHUNK_COLLECTION  = "chunks"

# bge models are asymmetric: passages are embedded as-is, but queries
# need an instruction prefix for best retrieval quality. Only used in
# embed_query() below, kept here so rag_search.py can import the same
# constant rather than risking the two getting out of sync.
EMBED_MODEL_NAME  = "BAAI/bge-base-en-v1.5"   # 768-dim, tuned for retrieval
EMBED_DIM         = 768
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CHUNK_SIZE      = 1200   # target characters per chunk
CHUNK_OVERLAP   = 200    # characters carried over between consecutive chunks
MIN_CHUNK_CHARS = 50     # drop near-empty fragments (e.g. trailing page headers)

EMBED_BATCH_SIZE = 64
MONGO_FLUSH_SIZE = 200    # how many chunk upserts to batch per bulk_write



# Chunking

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """
    Splits text into overlapping chunks, preferring paragraph boundaries
    (and sentence boundaries within an oversized paragraph) over hard
    character cuts, so each chunk stays semantically coherent rather than
    being sliced mid-sentence.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    current = ""

    def flush():
        nonlocal current
        buf = current.strip()
        if len(buf) >= MIN_CHUNK_CHARS:
            chunks.append(buf)

    for para in paragraphs:
        if len(para) > chunk_size:
            # Oversized paragraph (common in tariff/legal text run-ons) —
            # split on sentence boundaries instead.
            for sent in (s.strip() for s in para.replace("\n", " ").split(". ") if s.strip()):
                sent = sent if sent.endswith((".", "!", "?")) else sent + "."
                if len(current) + len(sent) + 1 <= chunk_size:
                    current = f"{current} {sent}".strip()
                else:
                    flush()
                    current = (current[-overlap:] + " " + sent).strip() if overlap else sent
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            flush()
            current = (current[-overlap:] + "\n\n" + para).strip() if overlap else para

    flush()
    return chunks


# ─────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[...] Loading embedding model '{EMBED_MODEL_NAME}'...")
        print("   First run can take a few minutes -- this pulls in torch/transformers")
        print("   (a big import chain) plus a one-time model download. This is normal,")
        print("   don't Ctrl+C; on Windows it's much faster if this project isn't inside")
        print("   a OneDrive/Dropbox-synced folder.")
        from sentence_transformers import SentenceTransformer  # lazy: avoid the heavy
        # torch/transformers import cost for any module that just needs constants
        # from this file (e.g. rag_search.py, app.py) without actually embedding anything.
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        print("[OK] Model loaded")
    return _model


def embed_passages(texts: list) -> list:
    """Embed a batch of document chunks. No instruction prefix — bge passages are embedded as-is."""
    model = get_model()
    vectors = model.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> list:
    """Embed a user question. bge models expect the instruction prefix on the query side only."""
    model = get_model()
    vector = model.encode(QUERY_INSTRUCTION + query, normalize_embeddings=True)
    return vector.tolist()


# ─────────────────────────────────────────────────────────────
# MongoDB
# ─────────────────────────────────────────────────────────────
def get_collections():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    db = client[DB_NAME]
    source = db[SOURCE_COLLECTION]
    chunks = db[CHUNK_COLLECTION]
    chunks.create_index("notification_id")
    chunks.create_index("chunk_id", unique=True)
    return source, chunks


def ensure_vector_index(chunks_col):
    """
    Attempts to create the Atlas Vector Search index via the driver.
    Programmatic Search Index management requires an Atlas tier that
    supports it — on the free/shared (M0) tier this call will fail,
    which is expected. In that case create the index once by hand
    (instructions printed below); it only needs to be done a single
    time, not on every pipeline run.
    """
    index_def = {
        "name": "vector_index",
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": EMBED_DIM, "similarity": "cosine"},
                {"type": "filter", "path": "category"},
            ]
        },
    }
    try:
        chunks_col.create_search_index(index_def)
        print("✅ Vector search index created via driver.")
    except Exception as e:
        print(f"⚠  Could not create the vector index automatically ({e}).")
        print_index_instructions()


def print_index_instructions():
    print("""
────────────────────────────────────────────────────────────
MANUAL STEP — create the Atlas Vector Search index (one-time only):
  1. Atlas UI → your cluster → Search → Create Search Index
  2. Choose "JSON Editor" → Database: regulai → Collection: chunks
  3. Index name: vector_index
  4. Paste this definition:
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 768, "similarity": "cosine" },
    { "type": "filter", "path": "category" }
  ]
}
  5. Save and wait ~1 minute for status to show "Active"
────────────────────────────────────────────────────────────
""")


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def run_pipeline():
    print("\n" + "=" * 60)
    print("  RegulAI — RAG Embedding Pipeline")      
    print("=" * 60)

    try:
        source, chunks_col = get_collections()
        print("✅ Connected to MongoDB Atlas")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return

    docs = list(source.find({"full_text": {"$exists": True, "$ne": ""}}))
    print(f"📄 {len(docs)} documents have full_text to embed")

    if not docs:
        print("⚠  Nothing to embed — run mongo_pipeline.py first to populate full_text.")
        return

    total_chunks = 0
    ops = []

    for doc in docs:
        notif_id = doc["notification_id"]
        pieces = chunk_text(doc.get("full_text", ""))
        if not pieces:
            continue

        vectors = embed_passages(pieces)

        for idx, (piece, vec) in enumerate(zip(pieces, vectors)):
            chunk_doc = {
                "chunk_id":        f"{notif_id}__{idx}",
                "notification_id": notif_id,
                "chunk_index":     idx,
                "text":            piece,
                "embedding":       vec,
                "title":           doc.get("title", ""),
                "category":        doc.get("category", ""),
                "notification_no": doc.get("notification_no", ""),
                "date":            doc.get("date", ""),
                "pdf_url":         doc.get("pdf_url", ""),
                "indexed_at":      datetime.utcnow().isoformat(),
            }
            ops.append(UpdateOne({"chunk_id": chunk_doc["chunk_id"]}, {"$set": chunk_doc}, upsert=True))
            total_chunks += 1

        if len(ops) >= MONGO_FLUSH_SIZE:
            _flush_ops(chunks_col, ops)
            ops = []

    if ops:
        _flush_ops(chunks_col, ops)

    print(f"\n✅ Embedded and upserted {total_chunks} chunks from {len(docs)} documents")
    ensure_vector_index(chunks_col)
    print("=" * 60 + "\n")


def _flush_ops(col, ops):
    try:
        result = col.bulk_write(ops, ordered=False)
        print(f"   ✅ Upserted batch: {result.upserted_count + result.modified_count} chunks")
    except BulkWriteError as bwe:
        print(f"   ⚠  Bulk write errors: {len(bwe.details.get('writeErrors', []))}")


if __name__ == "__main__":
    run_pipeline()








