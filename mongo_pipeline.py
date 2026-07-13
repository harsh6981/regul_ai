"""
mongo_pipeline.py — RegulAI MongoDB Integration
================================================
Reads cbic_master.json → extracts PDF text → upserts into MongoDB Atlas

Usage:
    python mongo_pipeline.py       

Requirements:
    pip install pymongo pymupdf python-dotenv
"""

import os
import json
import time
import fitz                          # PyMuPDF
from datetime import datetime
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIG — paste your MongoDB Atlas connection string below
# Get it from: Atlas → Connect → Drivers → copy the URI
# ─────────────────────────────────────────────────────────────
MONGO_URI    = os.getenv("MONGO_URI", "YOUR_MONGODB_ATLAS_URI_HERE")
DB_NAME      = "regulai"
COLLECTION   = "notifications"

META_FILE    = "cbic_master.json"
PDF_FOLDER = os.getenv("PDF_STORAGE_PATH", "CBIC_ALL_PDFS")

# How many characters of PDF text to store (None = all)
MAX_TEXT_CHARS = 50_000


# ─────────────────────────────────────────────────────────────
# PDF Text Extraction
# ─────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: str) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
    try:
        doc  = fitz.open(pdf_path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        if MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
        return text.strip()
    except Exception as e:
        print(f"  ⚠  PDF read error [{pdf_path}]: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# MongoDB helpers
# ─────────────────────────────────────────────────────────────
def get_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    # Ping to verify connection
    client.admin.command("ping")
    print("✅ Connected to MongoDB Atlas!!!!")
    db     = client[DB_NAME]
    col    = db[COLLECTION]

    # Create indexes for fast search
    col.create_index("notification_id", unique=True)
    col.create_index("category")
    col.create_index("date")
    col.create_index([("title", "text"), ("full_text", "text")])  # full-text search
    print(f"✅ Indexes ready on '{COLLECTION}'")
    return col


def build_upsert_ops(records: list) -> list:
    """Build bulk upsert operations — safe to re-run (idempotent)."""
    ops = []
    for rec in records:
        doc = {**rec}

        # Extract PDF text if not already present
        if not doc.get("full_text") and doc.get("file_location"):
            pdf_path = doc["file_location"]
            # Handle relative path
            if not os.path.isabs(pdf_path):
                pdf_path = os.path.join(PDF_FOLDER, os.path.basename(pdf_path))
            doc["full_text"] = extract_pdf_text(pdf_path)

        # Ensure keywords is a list
        if isinstance(doc.get("keywords"), str):
            doc["keywords"] = [k.strip() for k in doc["keywords"].split(",") if k.strip()]

        # Add pipeline timestamp
        doc["indexed_at"] = datetime.utcnow().isoformat()

        ops.append(UpdateOne(
            {"notification_id": doc["notification_id"]},   # match key
            {"$set": doc},
            upsert=True
        ))
    return ops


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def run_pipeline():
    print("\n" + "=" * 60)
    print("  RegulAI — MongoDB Indexing Pipeline")
    print("=" * 60)

    # 1. Load metadata
    if not os.path.exists(META_FILE):
        print(f"❌ {META_FILE} not found. Run project.py (scraper) first.")
        return

    with open(META_FILE, encoding="utf-8") as f:
        records = json.load(f)
    print(f"\n📄 Loaded {len(records)} records from {META_FILE}")

    # 2. Connect to MongoDB
    try:
        col = get_collection()
    except Exception as e:
        print(f" :( MongoDB connection failed: {e}")
        print("\n💡 Tip: Set MONGO_URI env var or edit MONGO_URI in this file.")
        return

    # 3. Process in batches of 100
    BATCH = 100
    total_upserted = 0
    total_matched  = 0

    for i in range(0, len(records), BATCH):
        batch   = records[i : i + BATCH]
        end_idx = min(i + BATCH, len(records))
        print(f"\n⚙  Processing records {i+1}–{end_idx}...")

        ops = build_upsert_ops(batch)

        try:
            result = col.bulk_write(ops, ordered=False)
            total_upserted += result.upserted_count
            total_matched  += result.matched_count
            print(f"   ✅ Upserted: {result.upserted_count}  |  Updated: {result.matched_count}")
        except BulkWriteError as bwe:
            print(f"   ⚠  Bulk write errors: {len(bwe.details.get('writeErrors', []))}")

        time.sleep(0.1)   # gentle on Atlas free tier

    # 4. Summary
    total_in_db = col.count_documents({})
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print(f"   New documents inserted : {total_upserted}")
    print(f"   Existing docs updated  : {total_matched}")
    print(f"   Total in MongoDB       : {total_in_db}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()