"""
duty_calculator/migrate_sqlite_to_mongo.py
=============================================
One-time migration: loads ICEDutyAI's bundled `dutyai_source.db` (SQLite —
1,192 HS codes from CUSTADA + 187 trade-defense/ADD-CVD-Safeguard measures)
into RegulAI's existing MongoDB `regulai` database, as two new collections:

    duty_hs_codes   — CUSTADA baseline rates per 8-digit HS code
    trade_defense   — anti-dumping / countervailing / safeguard measures

Named "duty_hs_codes" (not "hs_codes") to avoid any confusion with
RegulAI's existing `hsn_codes` collection (populated from CBIC tariff
PDFs via hsn_extract_pipeline.py) — the two are different data sources
covering the same HS-code space, kept as separate collections for now.
`duty_calculator/db.py` reads from `duty_hs_codes` / `trade_defense` only.

Safe to re-run: upserts by primary key (hs_8digit / measure_id), so
re-running just refreshes the data rather than duplicating it.

Usage:
    python duty_calculator/migrate_sqlite_to_mongo.py
"""
from __future__ import annotations

import os
import sqlite3
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "YOUR_MONGODB_ATLAS_URI_HERE")
DB_NAME = "regulai"

SQLITE_PATH = os.path.join(os.path.dirname(__file__), "dutyai_source.db")


def _sqlite_conn():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def migrate_hs_codes(mongo_db, sconn):
    col = mongo_db["duty_hs_codes"]
    col.create_index("hs_8digit", unique=True)
    col.create_index("chapter")

    rows = sconn.execute("SELECT * FROM hs_codes").fetchall()
    ops = []
    fixed_padding = 0
    for r in rows:
        doc = dict(r)
        # Source data bug: some hs_8digit values (mostly low chapter numbers
        # like chapter 1-9) have their leading zero stripped, e.g. "1010000"
        # instead of "01010000". Left-pad here so every key is a true 8-digit
        # string and exact-match lookups (db.py) work for every chapter.
        if len(doc["hs_8digit"]) < 8:
            doc["hs_8digit"] = doc["hs_8digit"].zfill(8)
            fixed_padding += 1
        ops.append(UpdateOne({"hs_8digit": doc["hs_8digit"]}, {"$set": doc}, upsert=True))

    if fixed_padding:
        print(f"  ℹ️  Fixed leading-zero padding on {fixed_padding} hs_8digit values")

    if not ops:
        print("  (no hs_codes rows found)")
        return 0

    try:
        result = col.bulk_write(ops, ordered=False)
        n = result.upserted_count + result.modified_count
        print(f"  ✅ duty_hs_codes: upserted/updated {n} of {len(rows)} rows")
        return n
    except BulkWriteError as bwe:
        print(f"  ⚠  Bulk write errors: {len(bwe.details.get('writeErrors', []))}")
        return 0


def migrate_trade_defense(mongo_db, sconn):
    col = mongo_db["trade_defense"]
    col.create_index("measure_id", unique=True)
    col.create_index([("hs_code_start", 1), ("hs_code_end", 1)])
    col.create_index("origin_country")

    rows = sconn.execute("SELECT * FROM trade_defense").fetchall()
    ops = []
    for r in rows:
        doc = dict(r)
        ops.append(UpdateOne({"measure_id": doc["measure_id"]}, {"$set": doc}, upsert=True))

    if not ops:
        print("  (no trade_defense rows found)")
        return 0

    try:
        result = col.bulk_write(ops, ordered=False)
        n = result.upserted_count + result.modified_count
        print(f"  ✅ trade_defense: upserted/updated {n} of {len(rows)} rows")
        return n
    except BulkWriteError as bwe:
        print(f"  ⚠  Bulk write errors: {len(bwe.details.get('writeErrors', []))}")
        return 0


def run():
    print("\n" + "=" * 60)
    print("  Duty Calculator — SQLite -> MongoDB migration")
    print("=" * 60)

    if not os.path.exists(SQLITE_PATH):
        print(f"❌ {SQLITE_PATH} not found. Copy ICEDutyAI's dutyai.db here first,")
        print(f"   named 'dutyai_source.db', then re-run.")
        return

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
        client.admin.command("ping")
        mongo_db = client[DB_NAME]
        print(f"✅ Connected to MongoDB Atlas ({DB_NAME})")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return

    sconn = _sqlite_conn()

    print("\n⚙  Migrating hs_codes -> duty_hs_codes ...")
    migrate_hs_codes(mongo_db, sconn)

    print("\n⚙  Migrating trade_defense -> trade_defense ...")
    migrate_trade_defense(mongo_db, sconn)

    sconn.close()

    print("\n" + "=" * 60)
    print("✅ MIGRATION COMPLETE")
    print(f"   duty_hs_codes total in Mongo : {mongo_db['duty_hs_codes'].count_documents({})}")
    print(f"   trade_defense total in Mongo : {mongo_db['trade_defense'].count_documents({})}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()