from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

c = MongoClient(os.getenv("MONGO_URI"))
db = c["regulai"]
col = db["hsn_codes"]
notif_col = db["notifications"]

# Dynamically find all distinct chapters that have been processed
processed_chapters = sorted(
    col.distinct("chapter"),
    key=lambda x: int(x) if x.isdigit() else 9999
)

# Also find all Tariff Schedule chapters in notifications to show unprocessed ones
all_tariff_docs = list(notif_col.find(
    {"category": "Tariff Schedule"},
    {"notification_id": 1, "hsn_extraction_complete": 1, "_id": 0}
))
all_chapters = sorted(
    {d["notification_id"].replace("chap-", "") for d in all_tariff_docs},
    key=lambda x: int(x) if x.isdigit() else 9999
)

print(f"\n{'='*70}")
print(f"  HSN Extraction Status — {len(processed_chapters)}/{len(all_chapters)} chapters processed")
print(f"{'='*70}")

for chapter in all_chapters:
    total = col.count_documents({"chapter": chapter})
    if total == 0:
        notif_doc = notif_col.find_one({"notification_id": f"chap-{chapter}"}, {"hsn_extraction_complete": 1})
        incomplete_flag = (notif_doc or {}).get("hsn_extraction_complete") is False
        incomplete_note = "  <-- INCOMPLETE (stopped mid-run)" if incomplete_flag else ""
        status = "NOT YET PROCESSED" + incomplete_note
        print(f"Chapter {chapter:>3}: {status}")
        continue

    rule    = col.count_documents({"chapter": chapter, "extraction_method": "rule_based"})
    cleaned = col.count_documents({"chapter": chapter, "extraction_method": "llm_cleanup"})

    notif_doc = notif_col.find_one({"notification_id": f"chap-{chapter}"}, {"hsn_extraction_complete": 1})
    incomplete_flag = (notif_doc or {}).get("hsn_extraction_complete") is False

    flagged_pct = (cleaned / total * 100) if total else 0
    flags = []
    if flagged_pct > 5:
        flags.append("[!] high LLM% -- spot-check rates")
    if incomplete_flag:
        flags.append("[INCOMPLETE] -- re-run this chapter")

    flag_str = "  " + " | ".join(flags) if flags else ""
    print(f"Chapter {chapter:>3}: {total:>4} rows | {rule:>4} rule_based | {cleaned:>3} llm_cleanup "
          f"({flagged_pct:.1f}%){flag_str}")

print(f"{'='*70}\n")
print(f"Unprocessed chapters: {[ch for ch in all_chapters if col.count_documents({'chapter': ch}) == 0]}\n")