from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

c = MongoClient(os.getenv("MONGO_URI"))
db = c["regulai"]
col = db["hsn_codes"]
notif_col = db["notifications"]

for chapter in ["28", "29", "30", "39", "41", "54", "55"]:
    total = col.count_documents({"chapter": chapter})
    rule = col.count_documents({"chapter": chapter, "extraction_method": "rule_based"})
    cleaned = col.count_documents({"chapter": chapter, "extraction_method": "llm_cleanup"})

    notif_doc = notif_col.find_one({"notification_id": f"chap-{chapter}"}, {"hsn_extraction_complete": 1})
    incomplete_flag = (notif_doc or {}).get("hsn_extraction_complete") is False

    if total == 0:
        print(f"Chapter {chapter}: NOT YET PROCESSED")
        continue
    flagged_pct = (cleaned / total * 100) if total else 0
    flag = "  <-- check this one" if flagged_pct > 5 else ""
    incomplete_note = "  <-- INCOMPLETE (quota ran out mid-chapter, re-run this one)" if incomplete_flag else ""
    print(f"Chapter {chapter}: {total} total | {rule} rule_based | {cleaned} llm_cleanup ({flagged_pct:.1f}%){flag}{incomplete_note}")