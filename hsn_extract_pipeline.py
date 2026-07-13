"""
hsn_extract_pipeline.py — RegulAI Structured HSN Code Extraction
==================================================================
Reads Tariff Schedule chapter PDFs (chap-1.pdf, chap-2.pdf, ...) page by
page and uses Gemini's structured-output mode to pull out individual HSN
code rows (code, description, unit, duty rate) into a new MongoDB
collection, 'hsn_codes' — enabling exact-match queries like "what is the
description for HSN code 390760" that full-text/vector search can't
answer reliably (a code has no semantic meaning for an embedding model
to latch onto, and the source PDFs are tabular, not prose).

Adapted from the page-by-page "read_books.py" pattern (PDF book →
knowledge base via OpenAI structured outputs), reworked for tabular
tariff data instead of unstructured book knowledge, and for Gemini's
OpenAI-compatible endpoint instead of native OpenAI. Confirmed via
Google's own docs that `client.beta.chat.completions.parse()` with a
Pydantic `response_format` works against Gemini's OpenAI-compat base_url
exactly as it does against native OpenAI — same SDK call, no fallback
needed.

Differences from the original script:
  - Schema is tabular rows (HSNRow), not a flat list of "knowledge points".
  - Storage is MongoDB ('hsn_codes' collection), not a local JSON file —
    but the same incremental-save-after-every-page resume pattern is kept,
    so a crash partway through a chapter doesn't lose already-extracted
    rows.
  - No "interval summary" / "final summary" feature — dropped entirely,
    not relevant to structured data extraction.
  - Carries the last (possibly incomplete) row from page N into the
    prompt for page N+1, since tariff tables can split a single row
    across a page boundary.

Usage:
    python hsn_extract_pipeline.py                  # process all Tariff Schedule chapters
    python hsn_extract_pipeline.py --chapter 39      # process just one chapter, e.g. for testing
    python hsn_extract_pipeline.py --test-pages 3    # only process the first N pages of each chapter (testing)

Requirements:
    pip install pymongo openai pymupdf pydantic python-dotenv
"""

from __future__ import annotations

import os
import re
import time
import argparse
from datetime import datetime, timezone
from typing import Optional

import fitz  # PyMuPDF
from pydantic import BaseModel
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from openai import OpenAI
from dotenv import load_dotenv

from rule_based_extract import parse_chapter_pdf as rule_based_parse

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MONGO_URI                = os.getenv("MONGO_URI", "YOUR_MONGODB_ATLAS_URI_HERE")
DB_NAME                  = "regulai"
NOTIFICATIONS_COLLECTION = "notifications"   # source: where chapter docs + file_location live
HSN_COLLECTION           = "hsn_codes"       # destination: structured per-code rows

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL    = "gemini-2.5-flash-lite"  # switched from 2.5-flash: same free-tier RPM-style pacing,
                                            # but a meaningfully higher daily request quota, and
                                            # well-suited to extraction/classification tasks like this one

PDF_FOLDER = "CBIC_ALL_PDFS"


# ─────────────────────────────────────────────────────────────
# Structured output schema
# ─────────────────────────────────────────────────────────────
class HSNRow(BaseModel):
    hsn_code: str        # e.g. "39072910" — string, not int, to preserve leading zeros / exact formatting
    description: str
    unit: Optional[str] = None             # e.g. "kg", "u", "-" if not specified
    standard_rate: Optional[str] = None    # "Rate of duty — Standard" column, e.g. "*7.5%", "Free"
    preferential_rate: Optional[str] = None  # "Rate of duty — Preferential Areas" column, e.g. "10%", or null if shown as "-"
    # Real CBIC tariff tables (confirmed against chap-29.pdf, chap-30.pdf) have BOTH
    # a Standard and a Preferential Areas duty column side by side — a single
    # `duty_rate` field would either drop one of them or jumble both into one
    # string. Kept as two separate optional fields instead.
    footnote_marker: Optional[str] = None
    # The literal marker symbol (e.g. "*", "**", "#") if standard_rate or
    # preferential_rate carries one, WITHOUT the marker baked into the rate
    # string itself — e.g. standard_rate="2.5%", footnote_marker="*", rather
    # than standard_rate="*2.5%". Kept separate so the rate value stays a
    # clean, directly-usable number/word, and so it can be joined against
    # this page's footnote legend to show *why* a marker is there (e.g. an
    # effective date) without re-parsing the rate string later.


class Footnote(BaseModel):
    marker: str          # the symbol exactly as printed, e.g. "*", "**", "#"
    text: str             # the footnote's own text, e.g. "w.e.f. 1.1.2022"


class PageExtraction(BaseModel):
    has_content: bool
    rows: list[HSNRow]
    footnotes: list[Footnote] = []
    # Footnote legends are printed once at the bottom of a page (e.g.
    # "* w.e.f. 1.1.2022", "#w.e.f.1.5.2023") and apply to every row on
    # that page carrying the matching marker. Captured separately here
    # rather than trying to inline the footnote text into each row during
    # extraction, since the legend can appear after every row it applies to.




# ─────────────────────────────────────────────────────────────
# Mongo
# ─────────────────────────────────────────────────────────────
_mongo_client = None


def get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
        _mongo_client.admin.command("ping")
    return _mongo_client[DB_NAME]


def get_notifications_collection():
    return get_db()[NOTIFICATIONS_COLLECTION]


def get_hsn_collection():
    col = get_db()[HSN_COLLECTION]
    col.create_index("hsn_code")
    col.create_index("chapter")
    return col


def _clean_rate(value: Optional[str]) -> Optional[str]:
    """
    Light normalization safety net for rate strings. Observed once in
    testing: the model occasionally emits a doubled "%%" (e.g. "*2.5%%"
    instead of "2.5%") — collapse that here rather than trusting every
    future extraction call to never repeat it.
    """
    if value is None:
        return None
    return re.sub(r'%%+', '%', value)


def _clean_marker(value: Optional[str]) -> Optional[str]:
    """
    Strips stray whitespace/newlines from a footnote marker. Observed in
    testing: the model occasionally emits "*\\n" instead of a clean "*",
    which silently breaks the footnote_lookup dict match (the row's
    marker no longer equals the legend's marker string exactly) even
    though both visually look like the same symbol. Without this, such
    rows get marker stored correctly-looking but footnote_text always
    resolves to None.
    """
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def upsert_hsn_rows(rows: list, chapter: str, source_pdf: str, footnotes: list = None) -> int:
    """
    Upserts extracted rows into 'hsn_codes'. Keyed on (hsn_code, source_pdf)
    rather than hsn_code alone — codes are mostly unique per chapter, but
    using the pair as the match key is cheap insurance against a stray
    cross-chapter collision silently overwriting a different chapter's row.

    footnotes (if given) is the page's list of Footnote(marker, text)
    entries extracted alongside these rows. Any row whose footnote_marker
    matches one of them gets the footnote's text attached directly as
    footnote_text, so a query can show "2.5% (w.e.f. 1.1.2022)" without
    needing a second lookup or having parsed the marker out of a rate
    string itself.

    Returns the number of rows upserted/updated.
    """
    if not rows:
        return 0
    footnote_lookup = {_clean_marker(fn.marker): fn.text for fn in (footnotes or [])}
    col = get_hsn_collection()
    ops = []
    for row in rows:
        marker = _clean_marker(row.footnote_marker)
        footnote_text = footnote_lookup.get(marker) if marker else None
        doc = {
            "hsn_code": row.hsn_code,
            "description": row.description,
            "unit": row.unit,
            "standard_rate": _clean_rate(row.standard_rate),
            "preferential_rate": _clean_rate(row.preferential_rate),
            "footnote_marker": marker,
            "footnote_text": footnote_text,
            "chapter": chapter,
            "source_pdf": source_pdf,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        ops.append(UpdateOne(
            {"hsn_code": row.hsn_code, "source_pdf": source_pdf},
            {"$set": doc},
            upsert=True,
        ))
    try:
        result = col.bulk_write(ops, ordered=False)
        return result.upserted_count + result.modified_count
    except BulkWriteError as bwe:
        print(f"   ⚠  Bulk write errors: {len(bwe.details.get('writeErrors', []))}")
        return 0


# ─────────────────────────────────────────────────────────────
# LLM extraction
# ─────────────────────────────────────────────────────────────
_llm_client = None


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file")
        _llm_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
    return _llm_client


EXTRACTION_SYSTEM_PROMPT = """You are extracting structured HSN (Harmonized System Nomenclature) tariff
code data from a page of an Indian Customs Tariff Schedule PDF.

The real table format (confirmed against actual CBIC chapter PDFs) has FIVE columns:
  Tariff Item | Description of goods | Unit | Rate of duty: Standard | Rate of duty: Preferential Areas
Capture the Standard and Preferential Areas rates as TWO SEPARATE fields (standard_rate,
preferential_rate) — do not merge them into one string, and do not drop the preferential
rate just because it's often shown as "-" (which means null/not applicable, not "same as
Standard").

SKIP the page (set has_content to false, rows to []) if it contains only:
- Disclaimers, copyright notices, or publishing details
- A bare chapter title/section divider page with no code table on it
- "Reserved for future use" placeholder pages
- Table of contents / index pages
- Chapter Notes / legal definitional text (numbered notes like "1. This Chapter does not
  cover...") with no tariff item table on the page
- Actual notification, exemption, or safeguard/countervailing-duty-order TEXT that happens
  to be printed at the end of a chapter PDF after the tariff table ends — these are real
  legal documents (look for phrases like "Notfn. No.", "G.S.R.", "Whereas, in the matter
  of...", "the Central Government hereby exempts/imposes...") and are NOT tariff code rows,
  even when they contain a small reference table of their own (e.g. a duty-rate table for a
  single safeguard order, or a "Patient Assistance Programme" table). Do not extract rows
  from these — they belong to a different part of the pipeline (notifications), not HSN
  code lookup.

IGNORE as pure noise, not data, wherever they appear on a page:
- The repeated column-header line "(1) (2) (3) (4) (5)"
- "SECTION-VI" / page-number / "CHAPTER-29" running headers at the top of the page

FOOTNOTES — capture these as structured data, AND link them to the rows that carry them:
- Footnote legend lines like "* w.e.f. 1.1.2022" / "** w.e.f. 1.5.2022" / "# w.e.f. 1.5.2023"
  usually appear together near the bottom of the page. For each one, add an entry to the
  `footnotes` list: {"marker": "*", "text": "w.e.f. 1.1.2022"} — marker is the exact symbol
  only (no rate value attached), text is the footnote's own explanation.
- CRITICAL — every row whose printed rate carries a marker symbol MUST have that row's
  footnote_marker field set to match. This is not optional and not rare: on a typical page,
  MOST or ALL rows carry a marker, because the marker indicates which version/date of the
  rate is currently in force. Do not default footnote_marker to null out of caution — check
  every single row's printed rate for a leading symbol and copy it into footnote_marker
  whenever one is present.

  Worked example — given this source text:
    2902 19 90
    --- Other
    kg.
    **2.5%
    -
    2902 20 00
    -
    Benzene
    kg.
    **2.5%
    -
  Both rows have "**" printed directly before their rate. The correct extraction is:
    {"hsn_code": "29021990", "description": "--- Other", "unit": "kg.",
     "standard_rate": "2.5%", "footnote_marker": "**", ...}
    {"hsn_code": "29022000", "description": "Benzene", "unit": "kg.",
     "standard_rate": "2.5%", "footnote_marker": "**", ...}
  Note that the rate is split clean ("2.5%") from the marker ("**") — NEITHER field should
  end up null when a marker is visibly printed on that row's rate in the source text.
- Only leave footnote_marker null when the row's rate genuinely has no symbol printed
  directly before it in the source text — check each row individually, don't assume.

ROWS THAT ARE NOT REAL DATA — handle these explicitly:
- A line consisting of just a tariff item code followed by the word "Omitted" (e.g.
  "2904 10 40 Omitted") means that code has been removed from the schedule and carries no
  description, unit, or rate. SKIP these — do not invent a description for them.
- A line of dashes ("- Other :", "- Unsaturated :", "-- Aromatic ethers...") with NO tariff
  item code in front of it is a CONTEXT HEADER that defines the category the following coded
  rows belong to. Do not create a separate row for it. However, you MUST inherit its text
  into every child row's description by joining: Parent Heading + ' - ' + Child Text (see
  DESCRIPTION INHERITANCE RULE below).

DESCRIPTION INHERITANCE RULE (mandatory — never output partial descriptions):
- Every HSN code's description must be FULLY SELF-CONTAINED. It must include all ancestor
  heading text (the 4-digit heading, then any one-dash sub-heading, then any two-dash
  sub-sub-heading, etc.), joined together with ' - '.
- Never store only a bare child fragment such as '-- Saturated', '-- Other', '--- Fresh',
  or any string that starts with '-'. A description starting with '-' is always WRONG.
- Always prepend all ancestor headings in order from outermost to innermost.

  Example A (one level of nesting):
    PDF:  2901  Acyclic hydrocarbons
          29011000  -- Saturated
          29012100  -- Ethylene
    Output descriptions:
      "Acyclic hydrocarbons - Saturated"   ✓
      "Acyclic hydrocarbons - Ethylene"    ✓
      "-- Saturated"                        ✗  (partial — WRONG)

  Example B (two levels of nesting):
    PDF:  1001  Wheat and meslin
          - Durum wheat :
          10011100  -- Seed
          10011900  -- Other
          - Other :
          10019010  -- Wheat for seed
          10019090  -- Other
    Output descriptions:
      "Wheat and meslin - Durum wheat - Seed"        ✓
      "Wheat and meslin - Durum wheat - Other"       ✓
      "Wheat and meslin - Other - Wheat for seed"    ✓
      "Wheat and meslin - Other - Other"             ✓

  Example C (three levels):
    PDF:  2902  Cyclic hydrocarbons
          - Cyclanes, cyclenes and cycloterpenes :
          -- Cyclohexane
          29021100  --- For use as power or heating fuels
    Output description:
      "Cyclic hydrocarbons - Cyclanes, cyclenes and cycloterpenes - Cyclohexane - For use as power or heating fuels"  ✓

EXTRACT a row for each genuine HSN code table entry you find. A row typically has:
- An HSN code (4, 6, or 8 digits, sometimes shown with spaces like "3907 29 10" — normalize
  to digits only, no spaces, e.g. "39072910")
- A description (the commodity name/description text for that code)
- A unit of quantity, if shown (e.g. "kg", "u", "-")
- Standard and Preferential Areas duty rates, if shown (e.g. "7.5%", "Free", "2.5%" — keep
  the rate value itself clean; see FOOTNOTES above for the marker, which is just as
  important to extract correctly as the rate itself)



IMPORTANT:
- Many rows have sub-codes / indented hierarchical entries (a heading code like "3907" with
  child codes "390710", "390720" etc. beneath it). Extract EVERY distinct code that has its
  own row in the table, including parent/heading rows if they have their own code and
  description, even if some fields (unit, rates) are blank for that row.
- If a code's description or rate appears to continue from the previous page (i.e. the
  page starts mid-row, with no visible code in the leftmost column for the first line), and
  you are given "carry-over context" describing the last incomplete row from the previous
  page, complete that row using the new page's content rather than creating an orphaned new
  row with a duplicate or guessed code.
- Real CBIC PDFs sometimes have minor column misalignment from text extraction (a rate
  value occasionally lands visually next to the wrong line). Use your judgment about which
  code a given rate belongs to based on table structure, but if it's genuinely ambiguous,
  leave the field null rather than guessing.
- Do not invent or guess a code, description, unit, or rate that is not actually present in
  the text. If a field genuinely isn't shown for a row, leave it null rather than guessing.
"""


def extract_page(client: OpenAI, page_text: str, page_num: int, carry_over: str = ""):
    """
    Sends one page of tariff PDF text to Gemini and returns a validated
    PageExtraction. carry_over (if non-empty) describes the last
    incomplete row from the previous page, so a row split across the
    page boundary can be completed rather than lost or duplicated.
    """
    user_content = f"Page text:\n{page_text}"
    if carry_over:
        user_content = f"Carry-over context (incomplete row from previous page): {carry_over}\n\n{user_content}"

    completion = client.beta.chat.completions.parse(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=PageExtraction,
    )
    return completion.choices[0].message.parsed


# Gemini's free tier caps gemini-2.5-flash at 5 requests/minute/project — firing
# through pages back-to-back blows through that by page 2-3 of any real chapter.
# MIN_SECONDS_BETWEEN_CALLS paces requests to stay under that even on a single
# pipeline instance; MAX_RETRIES + the backoff loop in extract_page_with_retry()
# handle the case where the quota's still hit anyway (e.g. another process is
# also calling the API), including parsing Google's own suggested retryDelay
# out of the 429 error body when present, rather than guessing a backoff blind.
MIN_SECONDS_BETWEEN_CALLS = 13   # 60s / 5 requests, plus a small margin
MAX_RETRIES = 4


def _parse_retry_delay_seconds(error_message: str) -> Optional[float]:
    """Gemini's 429 error body often includes a precise 'retryDelay': '29s'
    field — use it directly instead of guessing, when present."""
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(error_message))
    return float(m.group(1)) if m else None


def extract_page_with_retry(client: OpenAI, page_text: str, page_num: int, carry_over: str = ""):
    """
    Wraps extract_page() with retry-with-backoff for transient failures
    (429 rate-limit / 503 model-overloaded), which are common and expected
    on Gemini's free tier rather than real bugs. Re-raises the last error
    if all retries are exhausted, so the caller's existing "skip this page"
    handling still applies as a final fallback.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return extract_page(client, page_text, page_num, carry_over=carry_over)
        except Exception as e:
            last_error = e
            is_rate_or_overload = any(code in str(e) for code in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if not is_rate_or_overload or attempt == MAX_RETRIES - 1:
                raise
            delay = _parse_retry_delay_seconds(e) or (15 * (attempt + 1))  # 15s, 30s, 45s... if no hint given
            print(f"     ⏳ Page {page_num + 1}: rate-limited/overloaded, retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
    # pyrefly: ignore [bad-raise]
    raise last_error  # pragma: no cover — loop always returns or raises above


def looks_incomplete(row) -> bool:
    """Heuristic: a row missing unit and both rate fields is plausibly cut
    off mid-entry (e.g. the rates fell on the next page) rather than
    genuinely having no rate — worth carrying forward as context."""
    return row.unit is None and row.standard_rate is None and row.preferential_rate is None


# ─────────────────────────────────────────────────────────────
# HYBRID MODE — rule-based first pass (free, instant) + ONE batched
# LLM call per chapter to fix only the handful of rows the regex
# parser couldn't confidently resolve, instead of one LLM call per page.
# ─────────────────────────────────────────────────────────────
def rule_row_to_hsn_row(r: dict) -> HSNRow:
    return HSNRow(
        hsn_code=r['hsn_code'],
        description=r['description'],
        unit=r['unit'],
        standard_rate=_clean_rate(r['standard_rate']),
        preferential_rate=_clean_rate(r['preferential_rate']),
        footnote_marker=_clean_marker(r['footnote_marker']),
    )


def upsert_rule_based_rows(rows: list, chapter: str, source_pdf: str, extraction_method: str) -> int:
    """Same upsert shape as upsert_hsn_rows() but for plain dicts coming
    from the rule-based parser, which already has footnote_text resolved
    (no separate footnote_lookup join needed), and tags how each row was
    produced so you can audit/debug which path a given row came from."""
    if not rows:
        return 0
    col = get_hsn_collection()
    ops = []
    for r in rows:
        doc = {
            "hsn_code": r['hsn_code'],
            "description": r['description'],
            "unit": r['unit'],
            "standard_rate": _clean_rate(r.get('standard_rate')),
            "preferential_rate": _clean_rate(r.get('preferential_rate')),
            "footnote_marker": _clean_marker(r.get('footnote_marker')),
            "footnote_text": r.get('footnote_text'),
            "chapter": chapter,
            "source_pdf": source_pdf,
            "extraction_method": extraction_method,  # "rule_based" or "llm_cleanup"
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        ops.append(UpdateOne(
            {"hsn_code": doc["hsn_code"], "source_pdf": source_pdf},
            {"$set": doc},
            upsert=True,
        ))
    try:
        result = col.bulk_write(ops, ordered=False)
        return result.upserted_count + result.modified_count
    except BulkWriteError as bwe:
        print(f"   ⚠  Bulk write errors: {len(bwe.details.get('writeErrors', []))}")
        return 0


CLEANUP_SYSTEM_PROMPT = """You are fixing a small number of HSN tariff-table rows that a
rule-based parser flagged as low-confidence, by re-reading the relevant raw page text below.

For EACH flagged item, return one corrected row with: hsn_code (digits only, no spaces),
description, unit, standard_rate, preferential_rate, footnote_marker (the bare symbol only,
e.g. "*", or null if none). Use the SAME hsn_code values you were given — you're correcting
these specific rows, not finding new ones. If a flagged code genuinely has no real tariff
data (e.g. it turned out to be an "Omitted" entry or a section heading, not a real row),
omit it from your output rather than inventing data.

DESCRIPTION INHERITANCE RULE (mandatory):
Every description must be FULLY SELF-CONTAINED. Always prepend all ancestor heading text
(the 4-digit heading, then any dash-level sub-headings, in order), joined with ' - '.
Never output a description that starts with '-' or contains only a bare child fragment
like '-- Other' or '--- Fresh'. A description starting with '-' is always WRONG.

Example: for code 29011000 (-- Saturated) under heading '2901 Acyclic hydrocarbons',
the correct description is 'Acyclic hydrocarbons - Saturated', not '-- Saturated'."""


class CleanupRequest(BaseModel):
    rows: list[HSNRow]


def cleanup_flagged_rows_with_llm(client: OpenAI, doc, needs_review: list, chapter: str) -> list:
    """One batched call per chapter (not one per page) to fix only the rows
    rule_based_extract.py flagged. Returns a list of dicts in the same
    shape upsert_rule_based_rows() expects."""
    if not needs_review:
        return []

    pages_needed = sorted({r['page'] - 1 for r in needs_review if 'page' in r})
    context_text = "\n\n---PAGE BREAK---\n\n".join(
        f"[page {p + 1}]\n{doc[p].get_text()}" for p in pages_needed
    )
    flagged_summary = "\n".join(
        f"- hsn_code={r.get('hsn_code')!r}, page={r.get('page')}, issue={r.get('_flag')}, "
        f"known so far: description={r.get('description') or r.get('partial_description')!r}, "
        f"unit={r.get('unit') or r.get('partial_unit')!r}"
        for r in needs_review
    )
    user_content = (
        f"Flagged rows to fix:\n{flagged_summary}\n\n"
        f"Raw page text for context:\n{context_text}"
    )

    completion = client.beta.chat.completions.parse(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=CleanupRequest,
    )
    fixed = completion.choices[0].message.parsed.rows
    return [{
        'hsn_code': row.hsn_code,
        'description': row.description,
        'unit': row.unit,
        'standard_rate': row.standard_rate,
        'preferential_rate': row.preferential_rate,
        'footnote_marker': row.footnote_marker,
        'footnote_text': None,  # not re-resolved here; rare enough to check manually if it matters
    } for row in fixed]


# If the rule-based parser finds far fewer candidate rows than the page
# count plausibly accounts for, that's a sign the PDF doesn't match the
# layout it assumes (e.g. a two-column code/rate layout, or rate strings
# like "20% or Rs.115 per kg., whichever is higher" that its regexes don't
# recognize as a rate at all) — NOT that the chapter genuinely has few
# rows. Confirmed on chapter 54: 2 clean + 13 flagged = 15 candidates over
# 15 pages, vs. chapters 28-41's 15-60 rows/page. In that situation the
# LLM cleanup call can't help (it only repairs rows already found, it
# can't discover rows the parser never noticed), so trust is misplaced
# and it's safer to fall back to the slower-but-reliable per-page LLM path
# for that one chapter rather than upsert a near-empty result.
MIN_PLAUSIBLE_ROWS_PER_PAGE = 5.0


def rule_based_yield_looks_implausible(rows: list, needs_review: list, page_count: int) -> bool:
    if page_count == 0:
        return False
    candidates = len(rows) + len(needs_review)
    return (candidates / page_count) < MIN_PLAUSIBLE_ROWS_PER_PAGE


def process_chapter_pdf_hybrid(client: OpenAI, pdf_path: str, chapter: str, source_pdf: str,
                                use_llm_cleanup: bool = True, allow_auto_fallback: bool = True) -> dict:
    """
    Rule-based parse of the WHOLE chapter PDF (zero LLM calls), then — if
    use_llm_cleanup — exactly ONE LLM call for the entire chapter to fix
    whatever the regex parser flagged as low-confidence. This is the
    quota-friendly replacement for process_chapter_pdf() (which spends one
    LLM call per page).

    If the rule-based yield looks implausibly low for this PDF's page
    count (see rule_based_yield_looks_implausible), and allow_auto_fallback
    is True, this automatically defers to the full per-page LLM extraction
    instead of trusting/upserting the rule-based result — the cleanup call
    fixes flagged rows, it can't discover rows the parser missed entirely,
    so pushing a near-empty result through would be silently wrong rather
    than just incomplete.
    """
    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count
        rows, headings, needs_review, notification_text = rule_based_parse(doc)

        if allow_auto_fallback and rule_based_yield_looks_implausible(rows, needs_review, page_count):
            candidates = len(rows) + len(needs_review)
            print(f"  ⚠  {source_pdf}: only {candidates} candidate row(s) found across {page_count} page(s) "
                  f"({candidates / page_count:.1f}/page) — this doesn't look like a layout the rule-based "
                  f"parser handles correctly (e.g. multi-column PDF, or non-percentage rate text). "
                  f"Falling back to full per-page LLM extraction for this chapter instead of trusting it.")
            doc.close()
            fallback_result = process_chapter_pdf(client, pdf_path, chapter=chapter, source_pdf=source_pdf)
            if not fallback_result["complete"]:
                print(f"  ⚠  {source_pdf}: fallback LLM extraction stopped early at page "
                      f"{fallback_result['pages_processed']}/{fallback_result['pages_total']} "
                      f"(quota exhausted) — this chapter is INCOMPLETE, re-run it once quota resets.")
            return {
                "rule_based_rows": 0, "flagged": 0, "llm_fixed": 0,
                "total_upserted": fallback_result["total_upserted"],
                "auto_fallback_to_llm": True,
                "complete": fallback_result["complete"],
            }

        clean_rows = [r for r in rows if not r.get('_flag')]
        upserted = upsert_rule_based_rows(clean_rows, chapter, source_pdf, extraction_method="rule_based")
        print(f"  📄 {source_pdf}: {len(clean_rows)} row(s) parsed cleanly (0 LLM calls), "
              f"{len(needs_review)} flagged for cleanup")

        fixed_count = 0
        if use_llm_cleanup and needs_review:
            try:
                fixed_rows = cleanup_flagged_rows_with_llm(client, doc, needs_review, chapter)
                fixed_count = upsert_rule_based_rows(fixed_rows, chapter, source_pdf, extraction_method="llm_cleanup")
                print(f"     ✅ 1 batched LLM call fixed {fixed_count}/{len(needs_review)} flagged row(s)")
            except Exception as e:
                print(f"     ⚠  Cleanup LLM call failed ({e}) — {len(needs_review)} row(s) left unresolved")

        return {
            "rule_based_rows": len(clean_rows),
            "flagged": len(needs_review),
            "llm_fixed": fixed_count,
            "total_upserted": upserted + fixed_count,
            "auto_fallback_to_llm": False,
            "complete": True,
        }
    finally:
        if not doc.is_closed:
            doc.close()


# ─────────────────────────────────────────────────────────────
# Chapter-level orchestration
# ─────────────────────────────────────────────────────────────
def _is_quota_exhausted(e: Exception) -> bool:
    """
    True if this looks like a persistent, page-independent failure rather
    than a one-off transient blip — daily quota exhaustion (RESOURCE_
    EXHAUSTED/429/quota) OR a bad/invalid API key (401/UNAUTHENTICATED/
    ACCESS_TOKEN_TYPE_UNSUPPORTED — the recurring "pasted an OAuth token
    instead of a real GEMINI_API_KEY" mistake). extract_page_with_retry()
    already retries 429/503/RESOURCE_EXHAUSTED with backoff and only
    re-raises once retries are exhausted, but a 401 isn't retried at all
    (retrying with the same bad key just fails identically every time) —
    either way, once one page fails for one of these reasons, EVERY
    remaining page in the chapter will fail identically, so this is fatal
    for the whole run, not a "skip this one page and keep going" case.
    """
    return any(code in str(e) for code in (
        "RESOURCE_EXHAUSTED", "429", "quota", "QUOTA",
        "401", "UNAUTHENTICATED", "ACCESS_TOKEN_TYPE_UNSUPPORTED", "invalid authentication",
    ))


def process_chapter_pdf(client: OpenAI, pdf_path: str, chapter: str, source_pdf: str, test_pages=None) -> dict:
    """
    Processes one chapter PDF page by page, extracting and upserting HSN
    rows as it goes (so a crash partway through doesn't lose progress on
    earlier pages within the chapter).

    Returns a dict {"total_upserted", "complete", "pages_processed", "pages_total"}
    rather than a bare int — "complete" is the important addition: if quota
    is exhausted partway through, every remaining page would raise the same
    error, and silently "skip"-ing each one used to let this function return
    a small-but-real upsert count that looked indistinguishable from a
    genuinely short chapter (this is what happened to chapter 54: the
    fallback fired correctly, then quota ran out a couple of pages in, and
    the partial result was upserted and reported as if the chapter were
    done). Now a quota-exhaustion error aborts the remaining pages
    immediately and the caller is told the run is incomplete.
    """
    doc = fitz.open(pdf_path)
    pages_to_process = min(test_pages, doc.page_count) if test_pages else doc.page_count

    carry_over = ""
    total_upserted = 0
    last_call_time = 0.0
    pages_processed = 0
    complete = True

    print(f"  📄 {source_pdf}: {pages_to_process} page(s)")

    for page_num in range(pages_to_process):
        page_text = doc[page_num].get_text()
        if not page_text.strip():
            pages_processed += 1
            continue  # skip genuinely blank pages without spending an API call

        # Pace calls to stay under Gemini's free-tier 5-requests/minute cap.
        elapsed = time.time() - last_call_time
        if last_call_time and elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

        try:
            last_call_time = time.time()
            extraction = extract_page_with_retry(client, page_text, page_num, carry_over=carry_over)
        except Exception as e:
            if _is_quota_exhausted(e):
                print(f"     🛑 Page {page_num + 1}: persistent failure (quota exhausted or invalid "
                      f"GEMINI_API_KEY) ({e}) — aborting the rest of this chapter rather than reporting "
                      f"a false-complete partial result. Check your .env GEMINI_API_KEY (should start "
                      f"with 'AIzaSy', not be an OAuth token) or wait for quota to reset, then re-run "
                      f"this chapter.")
                complete = False
                break
            print(f"     ⚠  Page {page_num + 1}: extraction failed after retries ({e}) — skipping page")
            carry_over = ""
            pages_processed += 1
            continue

        pages_processed += 1

        if not extraction.has_content or not extraction.rows:
            print(f"     ·  Page {page_num + 1}: no table content (notes/disclaimer/etc.) — skipped")
            carry_over = ""
            continue

        upserted = upsert_hsn_rows(extraction.rows, chapter=chapter, source_pdf=source_pdf, footnotes=extraction.footnotes)
        total_upserted += upserted
        print(f"     ✅ Page {page_num + 1}: {len(extraction.rows)} row(s) extracted, {upserted} upserted")

        last_row = extraction.rows[-1]
        carry_over = (
            f"code={last_row.hsn_code!r}, description={last_row.description!r}, "
            f"unit={last_row.unit!r}, standard_rate={last_row.standard_rate!r}, "
            f"preferential_rate={last_row.preferential_rate!r}, footnote_marker={last_row.footnote_marker!r}"
            if looks_incomplete(last_row) else ""
        )

    doc.close()
    return {
        "total_upserted": total_upserted,
        "complete": complete,
        "pages_processed": pages_processed,
        "pages_total": pages_to_process,
    }


def run_pipeline(chapter_filter=None, test_pages=None, mode="hybrid"):
    print("\n" + "=" * 60)
    print("  RegulAI — HSN Code Structured Extraction Pipeline")
    print(f"  Mode: {mode}")
    print("=" * 60)

    try:
        notif_col = get_notifications_collection()
        print("✅ Connected to MongoDB Atlas")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return

    query = {"category": "Tariff Schedule"}
    if chapter_filter:
        query["notification_id"] = f"chap-{chapter_filter}"

    chapters = list(notif_col.find(query, {"_id": 0, "notification_id": 1, "file_location": 1, "title": 1}))
    print(f"📚 {len(chapters)} chapter document(s) to process")

    if not chapters:
        print("⚠  Nothing to process — check that mongo_pipeline.py has run and chapters exist.")
        return

    client = get_llm_client()
    grand_total = 0
    incomplete_chapters = []

    for chap_doc in chapters:
        notif_id = chap_doc["notification_id"]          # e.g. "chap-39"
        chapter_num = notif_id.replace("chap-", "")
        pdf_path = chap_doc.get("file_location", "")

        if not pdf_path or not os.path.exists(pdf_path):
            fallback = os.path.join(PDF_FOLDER, f"{notif_id}.pdf")
            if os.path.exists(fallback):
                pdf_path = fallback
            else:
                print(f"  ⚠  {notif_id}: PDF not found ({pdf_path or 'no path recorded'}) — skipping")
                continue

        if mode == "llm":
            result = process_chapter_pdf(client, pdf_path, chapter=chapter_num,
                                          source_pdf=f"{notif_id}.pdf", test_pages=test_pages)
        else:
            result = process_chapter_pdf_hybrid(client, pdf_path, chapter=chapter_num,
                                                 source_pdf=f"{notif_id}.pdf",
                                                 use_llm_cleanup=(mode == "hybrid"),
                                                 allow_auto_fallback=(mode == "hybrid"))

        upserted = result["total_upserted"]
        is_complete = result.get("complete", True)
        if not is_complete:
            incomplete_chapters.append(chapter_num)
        notif_col.update_one(
            {"notification_id": notif_id},
            {"$set": {"hsn_extraction_complete": is_complete}},
        )
        grand_total += upserted

    print(f"\n✅ Done — {grand_total} HSN row(s) upserted across {len(chapters)} chapter(s)")
    if incomplete_chapters:
        print(f"⚠  {len(incomplete_chapters)} chapter(s) stopped early due to quota exhaustion and are "
              f"INCOMPLETE — do not trust their row counts yet: {', '.join(incomplete_chapters)}")
        print(f"   Re-run with --chapter <N> for each of these once quota resets.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured HSN code rows from Tariff Schedule chapter PDFs")
    parser.add_argument("--chapter", type=str, default=None, help="Only process this chapter number, e.g. --chapter 39")
    parser.add_argument("--test-pages", type=int, default=None, help="Only process the first N pages of each chapter (for testing, --mode llm only)")
    parser.add_argument("--mode", choices=["hybrid", "rule", "llm"], default="hybrid",
                         help="hybrid (default): rule-based parse + 1 batched LLM cleanup call per chapter. "
                              "rule: rule-based only, zero LLM calls, leaves flagged rows unfixed. "
                              "llm: legacy per-page LLM extraction (one call per page).")
    args = parser.parse_args()

    run_pipeline(chapter_filter=args.chapter, test_pages=args.test_pages, mode=args.mode)