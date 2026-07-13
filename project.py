import requests
import pandas as pd
import base64
import os
import time
import re
from datetime import datetime
from typing import Union
import json
import sys
import io

# Fix for UnicodeEncodeError on Windows terminal when printing emojis
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


BASE_API   = "https://www.cbic.gov.in/api/cbic-content-msts/"
PDF_BASE   = "https://www.cbic.gov.in/content/pdf/"
PDF_FOLDER = os.getenv("PDF_STORAGE_PATH", "CBIC_ALL_PDFS")
META_FILE  = "cbic_master.json"
EXCEL_FILE = "cbic_master.xlsx"

os.makedirs(PDF_FOLDER, exist_ok=True)

visited = set()
rows = []

def detect_category(title: str) -> str:

    t = title.lower()
    # Most specific / narrowest-signal categories first. These all get
    # swallowed by the generic "duty" or "notification" checks if those
    # run earlier, since this is a first-match chain.
    if any(k in t for k in ["anti-dumping", "antidumping", "dumping"]): return "Anti-Dumping"
    if any(k in t for k in ["safeguard"]):                              return "Safeguard"
    if any(k in t for k in ["countervailing", "cvd"]):                  return "Countervailing Duty"
    if any(k in t for k in ["drawback"]):                               return "Drawback"
    if any(k in t for k in ["gst", "igst", "cgst", "sgst"]):            return "GST"
    if any(k in t for k in ["exemption", "exempt"]):                    return "Exemption"
    if any(k in t for k in ["trade notice", "trade policy"]):           return "Trade Notice"
    if any(k in t for k in ["import", "imports"]):                      return "Import Policy"
    if any(k in t for k in ["export", "exports"]):                      return "Export Policy"
    # Broad catch-alls last, since almost every notification mentions
    # "duty" or "tariff" somewhere.
    if any(k in t for k in ["customs duty", "tariff", "bcd", "duty"]):  return "Customs Duty"
    if any(k in t for k in ["circular"]):                               return "Circular"
    if any(k in t for k in ["notification", "notfn"]):                  return "Notification"
    return "General"
   


def is_tariff_schedule_doc(file_path: str) -> bool:
    """
    Detects documents belonging to the static Customs Tariff Schedule
    (the HSN classification book — chapter listings like 'Live animals',
    'Cereals', etc.) as opposed to actual notifications/circulars.
    These have no notification number or date and no policy content in
    their titles, so keyword-based detect_category() can never classify
    them correctly — they need to be identified by their file path
    instead, e.g. .../Customs/Tariff/.../CUSTOMS_TARIFF_VOL-I/chap-1.pdf
    """
    p = file_path.lower()
    return "customs_tariff" in p or "/tariff/" in p


KEYWORD_LIST = [
    "polymer", "chemical", "steel", "textile", "pharmaceutical",
    "electronics", "rubber", "plastic", "fertilizer", "petroleum",
    "machinery", "automobile", "garment", "leather", "ceramic",
    "duty", "tariff", "restriction", "prohibition", "exemption",
    "license", "permit", "quota", "drawback", "refund",
    "import", "export", "customs", "gst", "igst", "cgst", "sgst",
    "anti-dumping", "safeguard", "countervailing",
    "notification", "circular", "amendment", "superseded",
]

def extract_keywords(text: str) -> list:
    t = text.lower()
    return [k for k in KEYWORD_LIST if k in t]


# ── Notification number extraction ───────────────────────────────────────────
def extract_notif_number(title: str, doc_data: dict) -> str:
    # Try API fields first
    for field in ["notificationNo", "circularNo", "orderNo", "docNo", "refNo"]:
        val = doc_data.get(field, "")
        if val:
            return str(val).strip()

    # Try regex from title
    patterns = [
        r'notification\s+no\.?\s*([\w\/\-]+)',
        r'circular\s+no\.?\s*([\w\/\-]+)',
        r'no\.\s*([\d\/\-]+)',
        r'(\d+\/\d{4})',
        r'([\d]+\/[\w]+\/[\d]+)',
    ]
    for p in patterns:
        m = re.search(p, title, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return "N/A"


# ── Date parsing ──────────────────────────────────────────────────────────────
def parse_date(doc_data: dict) -> str:
    for field in ["issueDt", "docDate", "notifDate", "date", "effectiveDt"]:
        val = doc_data.get(field, "")
        if val:
            # Try to parse and standardize
            for fmt in ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d %b %Y", "%d-%b-%Y"]:
                try:
                    return datetime.strptime(str(val).strip(), fmt).strftime("%d-%m-%Y")
                except Exception:
                    continue
            return str(val).strip()
    return "N/A"


# ── PDF downloader ────────────────────────────────────────────────────────────
def save_pdf(pdf_api: str, filename: str) -> Union[str, bool]:
    try:
        response = requests.get(pdf_api, timeout=60)
        if response.status_code != 200:
            return False

        data = response.json()
        if not data.get("data"):
            return False

        pdf_bytes = base64.b64decode(data["data"])
        filepath  = os.path.join(PDF_FOLDER, filename)

        with open(filepath, "wb") as f:
            f.write(pdf_bytes)

        return filepath

    except Exception as e:
        print(f"  ⚠ PDF Error [{filename}]: {e}")
        return False


# ── Page crawler ────────────────────────────────────────────────────
def crawl_page(page_id: str, parent_title: str = "", depth: int = 0):
    if page_id in visited:
        return
    visited.add(page_id)

    url = BASE_API + page_id
    try:
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            return
        data = response.json()
    except Exception as e:
        print(f"   Page Error [{page_id}]: {e}")
        return

    current_title = data.get("titleEn", parent_title).strip()
    indent        = "  " * depth

    print(f"{indent}📂 {current_title}")

    # ── Process documents on this page ──
    for doc in data.get("cbicDocMsts", []):
        try:
            file_path = doc.get("filePathEn", "")
            if not file_path:
                continue

            filename   = file_path.split("/")[-1]
            pdf_api    = PDF_BASE + file_path
            doc_title  = doc.get("docTitleEn", "").strip()

            # Extract all metadata
            notif_num  = extract_notif_number(doc_title or current_title, doc)
            date       = parse_date(doc)
            if is_tariff_schedule_doc(file_path):
                category = "Tariff Schedule"
            else:
                category = detect_category(f"{doc_title} {current_title}")
            keywords   = extract_keywords(doc_title + " " + current_title)
            source_url = f"https://www.cbic.gov.in/content/pdf/{file_path}"

            # Download PDF
            saved_file = save_pdf(pdf_api, filename)

            record = {
                # ── Core fields (match project spec) ──
                "notification_id":  filename.replace(".pdf", ""),
                "notification_no":  notif_num,
                "title":            doc_title or current_title,
                "date":             date,
                "authority":        "CBIC",
                "category":         category,
                "keywords":         keywords,
                "summary":          "",           # filled by pipeline later
                "full_text":        "",           # filled by pipeline later
                "pdf_url":          source_url,
                "file_location":    saved_file or "",
                "related_notifications": [],      # filled by pipeline later
                "embedding":        [],           # filled by pipeline later

                # ── Extra scraper fields ──
                "page_title":       current_title,
                "raw_doc_title":    doc_title,
                "pdf_api_url":      pdf_api,
                "scraped_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            } 

            rows.append(record)
            print(f"{indent}  ✅ {notif_num} | {doc_title[:50] or filename}")

        except Exception as e:
            print(f"{indent}  ⚠    Document Error: {e}")

    # ── Recurse into child pages ──
    for child in data.get("childContentList", []):
        path = child.get("path", "")
        if not path:
            continue
        child_id = path.split("/")[-1]
        crawl_page(child_id, current_title, depth + 1)
        time.sleep(0.3)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global visited, rows
    visited = set()
    rows    = []

    print("Starting Scraper")
    print(f"📁 PDFs → {PDF_FOLDER}/")
    print(f"📊 Metadata → {EXCEL_FILE} + {META_FILE}")
    print("-" * 60)

    try:
        root = requests.get(BASE_API + "Njk=", timeout=60).json()
    except Exception as e:
        print(f"❌ Failed to fetch root: {e}")
        return

    for tariff in root.get("childContentList", []):
        try:
            path      = tariff.get("path", "")
            tariff_id = path.split("/")[-1]
            crawl_page(tariff_id)
        except Exception as e:
            print(f"⚠ Tariff Error: {e}")

    # ── Save Excel ──
    if not rows:
        print("⚠  No records scraped — skipping file save.")
        return
    df = pd.DataFrame(rows)

    # Drop heavy fields for Excel (keep in JSON)
    excel_cols = [
        "notification_id", "notification_no", "title", "date",
        "authority", "category", "keywords", "pdf_url",
        "file_location", "page_title", "scraped_at"
    ]
    df_excel = df[[c for c in excel_cols if c in df.columns]].copy()
    df_excel["keywords"] = df_excel["keywords"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )
    df_excel.to_excel(EXCEL_FILE, index=False)
    print(f"\n📊 Excel saved → {EXCEL_FILE}")

    # ── Save full JSON (includes all fields for MongoDB pipeline) ──
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"📄 JSON saved → {META_FILE}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("✅ SCRAPING COMPLETE")
    print(f"   Total Documents : {len(rows)}")
    print(f"   PDFs Downloaded : {sum(1 for r in rows if r['file_location'])}")
    print(f"   PDFs Failed     : {sum(1 for r in rows if not r['file_location'])}")
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print("\n   By Category:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"     {cat:<25} {count}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠  Interrupted! Saving partial data...")
        if rows:
            import json
            with open("cbic_master.json", "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(rows)} records to cbic_master.json")
        else:
            print("  No records to save.")