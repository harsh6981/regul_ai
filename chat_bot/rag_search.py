"""
rag_search.py — RegulAI Retrieval-Augmented Generation
========================================================
Given a user question:
  1. Embed it with the same Sentence Transformers model used for
     indexing (embed_pipeline.py) so query and passage vectors live
     in the same space.
  2. Run $vectorSearch against the 'chunks' collection in Atlas to
     pull back the most relevant passages.
  3. Feed those passages + the question to Gemini (2.5 Flash, free
     tier, 1M token context) to generate a grounded, cited answer.

     NOTE on SDK: this used to go through Gemini's OpenAI-compatible
     endpoint via the `openai` package, with `thinking_config` sent by
     hand as a guessed `extra_body` JSON shape. That guessing produced
     a real 400 ("found both reasoning_effort and thinking_config") and,
     even once that was fixed, gave no real visibility into whether
     thinking tokens were actually being suppressed — the compat layer
     never reports `thoughts_token_count`, so a suspiciously small
     completion (e.g. 70 tokens with finish_reason=length) couldn't be
     diagnosed with confidence. This now uses Google's native
     `google-genai` SDK instead, where `thinking_config` is a first-class,
     documented `GenerateContentConfig` field — no shape-guessing — and
     `usage_metadata` reports `thoughts_token_count` separately, so we can
     actually confirm thinking is off rather than infer it.

Import answer_question() from app.py's /api/chat (or similar) route.

Requirements:
    pip install pymongo google-genai python-dotenv
    (sentence-transformers is pulled in via embed_pipeline)
"""

import os
import re
from pymongo import MongoClient
from google import genai
from google.genai import types
from dotenv import load_dotenv

from embed_pipeline import embed_query, DB_NAME, CHUNK_COLLECTION, MONGO_URI

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL    = "gemini-2.5-flash"   # free tier, no card required, 1M context
VECTOR_INDEX    = "vector_index"
NOTIFICATIONS_COLLECTION = "notifications"

_mongo_client = None
_genai_client = None


def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file")
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def call_gemini(messages: list, system_prompt: str, max_tokens: int, temperature: float = 0.3,
                 model: str = GEMINI_MODEL, thinking_budget: int = 0) -> dict:
    """
    Shared Gemini call point — used by both the single-entity and
    multi-entity (comparison) paths below, and by app.py's /api/chat
    route, so there is exactly one place that knows how to talk to
    Gemini and one place to fix if the SDK shape ever changes again.

    `messages` should already be role-normalized user/assistant turns
    (see trim_history()) — system_prompt is passed separately since the
    native SDK takes it as its own config field rather than a message.

    thinking_budget: 0 disables thinking entirely (cheapest, fine for
    simple lookups/templated answers). For harder synthesis tasks, prefer
    an explicit positive budget (e.g. 1024) over -1 (dynamic) — dynamic
    thinking is documented as unreliable in combination with
    max_output_tokens on gemini-2.5-flash: thoughts_token_count counts
    against max_output_tokens but isn't properly reconciled against it,
    so the model can hit MAX_TOKENS at wildly different, much-smaller-
    than-configured totals from run to run (confirmed here: one run cut
    off at ~70 combined tokens, another at ~1081, both with max_output_
    tokens set far higher). An explicit budget plus generous headroom
    above it is the documented workaround.

    Returns a dict: {"text", "finish_reason", "usage"} — finish_reason
    is normalized to a plain string like "STOP" / "MAX_TOKENS" so callers
    don't need to know about the SDK's enum type.
    """
    client = get_genai_client()

    contents = [
        types.Content(
            role=("user" if msg["role"] == "user" else "model"),
            parts=[types.Part(text=msg["content"])],
        )
        for msg in messages
    ]

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )

    response = client.models.generate_content(model=model, contents=contents, config=config)

    text = response.text or ""

    finish_reason = None
    try:
        finish_reason = response.candidates[0].finish_reason
        finish_reason = getattr(finish_reason, "name", finish_reason)  # enum -> str
    except (AttributeError, IndexError):
        pass

    usage = getattr(response, "usage_metadata", None)

    return {"text": text, "finish_reason": finish_reason, "usage": usage}


def get_chunks_collection():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client[DB_NAME][CHUNK_COLLECTION]


def get_notifications_collection():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client[DB_NAME][NOTIFICATIONS_COLLECTION]


# ─────────────────────────────────────────────────────────────
# Notification-number direct lookup
# ─────────────────────────────────────────────────────────────
# Same idea as the chapter-number lookup: vector search struggles to
# connect a bare alphanumeric code like "50/2017" to the right document
# on meaning alone, so we try an exact/regex match against the
# 'notification_no' field first and only fall back to vector search if
# nothing is found. Patterns cover the formats extract_notif_number()
# (project.py) actually produces: "50/2017", "50/2017-Cus",
# "12/2023-Customs (N.T.)", etc.
NOTIF_NUMBER_PATTERNS = [
    r'notification\s+(?:no\.?|number)?\s*([\d]+\/[\d]{4}[\w\-\.\(\)\/]*)',
    r'\bno\.?\s*([\d]+\/[\d]{4}[\w\-\.\(\)\/]*)',
    r'\b(\d{1,5}\/\d{4}(?:[\-\/][A-Za-z\.\(\) ]+)?)\b',
]


def extract_notification_number(query: str):
    """
    Pulls a notification-number-shaped token out of a free-text question,
    e.g. "tell me about notification 50/2017" -> "50/2017". Returns None
    if nothing matching the pattern is found.
    """
    for pattern in NOTIF_NUMBER_PATTERNS:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".,;:")
    return None


def lookup_by_notification_number(notif_no: str):
    """
    Exact-match lookup against the 'notifications' collection. Tries an
    exact (case-insensitive) match first; if that fails, tries a loose
    match on just the leading "NN/YYYY" part, since real-world citations
    are inconsistent about trailing suffixes like "-Cus" or "(N.T.)".
    Returns the full document (including full_text) or None.

    Uses re.compile() (PyMongo translates this to a BSON regex) rather
    than the dict-style {"$regex": "...", "$options": "i"} operator —
    confirmed by direct testing that the dict form silently matches zero
    documents on this setup (PyMongo/Atlas version-specific quirk) while
    re.compile() works correctly against the exact same field.
    """
    col = get_notifications_collection()

    pattern = re.compile(f"^{re.escape(notif_no)}$", re.IGNORECASE)
    doc = col.find_one({"notification_no": pattern}, {"_id": 0})
    if doc:
        return doc

    # Loosen to just the numeric/year prefix, e.g. "50/2017" out of "50/2017-Cus"
    prefix_match = re.match(r'^(\d+\/\d{4})', notif_no)
    if prefix_match:
        prefix = prefix_match.group(1)
        prefix_pattern = re.compile(f"^{re.escape(prefix)}", re.IGNORECASE)
        doc = col.find_one({"notification_no": prefix_pattern}, {"_id": 0})
        if doc:
            return doc

    return None


# ─────────────────────────────────────────────────────────────
# Chapter-number direct lookup
# ─────────────────────────────────────────────────────────────
# Tariff Schedule chapters (chap-1.pdf, chap-2.pdf, ... see project.py /
# is_tariff_schedule_doc()) have no notification number and no date —
# their only identifying field is notification_id, set straight from the
# filename: "chap-2". A bare chapter number has zero semantic content
# (chapter 2 isn't "closer" in meaning to chapter 4 than to chapter 84),
# so vector search has no real signal to work with and effectively
# returns near-random chapters. Same fix as notification-number lookup:
# bypass vector search entirely when the question names a chapter.
CHAPTER_NUMBER_PATTERNS = [
    r'\bchapter\s*\.?\s*(\d{1,2})\b',
    r'\bchap\s*\.?\s*(\d{1,2})\b',
    r'\bhsn\s+(?:code\s+)?(?:for\s+)?chapter\s*(\d{1,2})\b',
]


def extract_chapter_number(query: str):
    """
    Pulls a single chapter number out of a free-text question (the FIRST
    one found), e.g. "hsn code for chapter 2" -> "2". Returns None if no
    chapter-shaped reference is found. Kept for backwards-compat
    (followup resolution, single-chapter lookups) — use
    extract_chapter_numbers() (plural) for multi-chapter questions like
    comparisons ("compare chapter 29 and 39").
    """
    chapters = extract_chapter_numbers(query)
    return chapters[0] if chapters else None


def extract_chapter_numbers(query: str):
    """
    Pulls ALL chapter numbers out of a free-text question, e.g.
    "compare chapter 29 and chapter 39" -> ["29", "39"]. Also handles
    the more natural phrasing where "chapter"/"chap" is only said once
    and the second number is bare, e.g. "compare chapter 29 and 39" or
    "chapter 29 vs 39" -> ["29", "39"]. The bare-number half of that
    only matches when it immediately follows an already-matched chapter
    mention via a connector word (and/&/vs/versus/,/to) — this is
    deliberately narrow so a sentence like "chapter 29 has 40 headings"
    doesn't spuriously pick up "40" as a second chapter.
    Dedupes while preserving order.
    """
    found = []
    seen_spans = []

    def overlaps(span):
        return any(s[0] < span[1] and span[0] < s[1] for s in seen_spans)

    for pattern in CHAPTER_NUMBER_PATTERNS:
        for m in re.finditer(pattern, query, re.IGNORECASE):
            span = m.span(1)
            if overlaps(span):
                continue
            seen_spans.append(span)
            num = m.group(1).strip()
            if num not in found:
                found.append(num)

    # Second pass: catch a bare trailing number joined to an already-found
    # chapter mention by a connector ("and 39", "& 39", "vs 39", "to 39",
    # ", 39"). Anchored right after the END of a previously matched span
    # so it can't drift onto an unrelated number later in the sentence.
    trailing_re = re.compile(r'\s*(?:and|&|vs\.?|versus|to|,)\s*(\d{1,2})\b', re.IGNORECASE)
    for span in list(seen_spans):
        m = trailing_re.match(query, span[1])
        if m:
            extra_span = m.span(1)
            if overlaps(extra_span):
                continue
            seen_spans.append(extra_span)
            num = m.group(1).strip()
            if num not in found:
                found.append(num)

    return found


def lookup_by_chapter_number(chapter_no: str):
    """
    Exact-match lookup against the 'notifications' collection by
    notification_id, e.g. "2" -> "chap-2". Only matches documents tagged
    "Tariff Schedule" so a coincidental "chap-N" id outside that category
    (unlikely, but not impossible) can't be mismatched. Returns the full
    document (including full_text) or None.

    Uses re.compile() rather than dict-style $regex — see
    lookup_by_notification_number() docstring for why.
    """
    col = get_notifications_collection()
    pattern = re.compile(f"^chap-{re.escape(chapter_no)}$", re.IGNORECASE)
    return col.find_one(
        {"notification_id": pattern, "category": "Tariff Schedule"},
        {"_id": 0},
    )


# ─────────────────────────────────────────────────────────────
# Conversational follow-up resolution (generalized)
# ─────────────────────────────────────────────────────────────
# The original version of this only handled bare-number chapter follow-ups
# ("chapter 39" then just "26"). The actual bug reported in practice was
# broader: after looking up HSN code 2923 20 90 by code, a follow-up like
# "what is description" or "give me more info on that" contains no code of
# its own, so extract_hsn_code() returns None on it, the direct-lookup path
# never fires, and the question falls through to vector search — which
# has no real signal for a code-less follow-up and surfaces unrelated
# chunks (hence "the documents do not contain that information" answers,
# even though the row demonstrably existed). Fix: when the current
# message doesn't contain an extractable entity of its own AND looks like
# a vague/short follow-up, reuse the most recent entity from earlier in
# the conversation instead of going straight to vector search.
FOLLOWUP_HINT_RE = re.compile(
    r'\b(more|info|information|description|desc|details?|this|that|it|again|'
    r'rate|duty|unit|note|footnote)\b', re.IGNORECASE
)


def looks_like_vague_followup(query: str) -> bool:
    """
    True if this message is short/generic enough that it's very unlikely
    to be a fresh, self-contained question — "what is description", "give
    me more info on that", "and this one" — rather than a new topic.
    """
    words = query.strip().split()
    if not words:
        return False
    if len(words) <= 6:
        return True
    return bool(FOLLOWUP_HINT_RE.search(query)) and len(words) <= 14


def find_recent_entity_followup(messages: list):
    """
    Scans backwards through prior user turns (excluding the latest, which
    the caller already tried and failed to extract anything from) for the
    most recent message containing an extractable HSN code, notification
    number, or chapter number. Returns (kind, value) for the first one
    found, or (None, None) if nothing turns up in recent history.
    """
    for m in reversed(messages[:-1]):
        if m.get("role") != "user":
            continue
        text = m.get("content", "")
        hsn = extract_hsn_code(text)
        if hsn:
            return ("hsn_code", hsn)
        notif = extract_notification_number(text)
        if notif:
            return ("notification_number", notif)
        chap = extract_chapter_number(text)
        if chap:
            return ("chapter_number", chap)
    return (None, None)


def find_recent_multi_entity_followup(messages: list):
    """
    Scans backwards through prior user turns (excluding the latest) for
    the most recent message that named 2+ HSN codes OR 2+ chapter
    numbers, and returns (kind, [values]) for that message — or
    (None, None) if no prior message had 2+ of the same kind.

    This is deliberately NOT gated on specific wording like "both" or
    "each" — a follow-up that should reuse a previously-established
    multi-entity set takes many forms that don't share any common
    keyword: "make it in tabular form", "compare duty rate of both the
    chapters", "what about each of them", "format that as a table". The
    common thread isn't the wording, it's that find_recent_entity_followup
    (singular) would otherwise only recover ONE of the entities, silently
    dropping the rest — which is what was producing exhaustive single-
    entity dumps and MAX_TOKENS cutoffs on requests that were really
    asking to reformat/re-examine a set of 2+ things already on the table.
    Caller is responsible for checking looks_like_vague_followup() first
    so this doesn't get applied to genuinely new, self-contained questions.
    """
    for m in reversed(messages[:-1]):
        if m.get("role") != "user":
            continue
        text = m.get("content", "")
        hsn_codes = extract_hsn_codes(text)
        if len(hsn_codes) >= 2:
            return ("hsn_code", hsn_codes)
        chapter_nos = extract_chapter_numbers(text)
        if len(chapter_nos) >= 2:
            return ("chapter_number", chapter_nos)
    return (None, None)


# ─────────────────────────────────────────────────────────────
# HSN-code direct lookup
# ─────────────────────────────────────────────────────────────
# Structured rows produced by hsn_extract_pipeline.py, stored in a
# separate 'hsn_codes' collection (one document per code: hsn_code,
# description, unit, duty_rate, chapter, source_pdf) rather than relying
# on vector search over raw chapter text. An HSN code like "390760" has
# no semantic content for an embedding model and the source PDFs are
# tabular, not prose, so this is the same rationale as notification- and
# chapter-number lookup: bypass vector search entirely for these queries.
HSN_COLLECTION = "hsn_codes"

# HSN codes are 4, 6, or 8 digits — sometimes written with spaces
# ("3907 29 10") or the word "code"/"HSN" preceding them. Require at
# least 4 digits so this doesn't collide with the 1-2 digit chapter
# number pattern above. Each alternative explicitly allows optional
# spaces between digit groups (\d{4,8} alone stops at the first space
# due to \b, which previously truncated "3907 29 10" to just "3907").
HSN_CODE_PATTERNS = [
    r'\bhsn\s*(?:code)?\s*(?:for)?\s*[:\-]?\s*(\d{4}(?:\s?\d{2}){0,2})\b',
    r'\bcode\s*(\d{4}(?:\s?\d{2}){0,2})\b',
    r'\b(\d{4}\s\d{2}\s\d{2})\b',   # bare spaced 8-digit, e.g. "3907 29 10"
    r'\b(\d{4}\s\d{2})\b',          # bare spaced 6-digit heading, e.g. "3904 30"
    r'\b(\d{6})\b',                 # bare unspaced 6-digit
    r'\b(\d{8})\b',                 # bare unspaced 8-digit
]


def get_hsn_collection():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client[DB_NAME][HSN_COLLECTION]


def extract_hsn_code(query: str):
    """
    Pulls a single HSN-code-shaped token out of a free-text question
    (the FIRST one found, in pattern-priority order), e.g. "what is
    description for 390760" -> "390760". Returns None if nothing
    matches. Kept for backwards-compat (followup resolution, simple
    single-code lookups) — use extract_hsn_codes() (plural) for
    multi-code questions like comparisons.
    """
    codes = extract_hsn_codes(query)
    return codes[0] if codes else None


def extract_hsn_codes(query: str):
    """
    Pulls ALL HSN-code-shaped tokens out of a free-text question, e.g.
    "compare 2923 20 90 with 2933 39 90" -> ["29232090", "29333990"].
    Needed so multi-code questions (comparisons, "X vs Y", "and also Z")
    don't silently collapse to just the first code mentioned. Dedupes
    while preserving order. Deliberately requires 4+ digits so this
    doesn't fire on a bare 1-2 digit chapter reference.

    Patterns are tried in priority order across the WHOLE query (not
    just up to the first match), and each pattern is matched with
    finditer so e.g. two "hsn code ..." mentions in one message both
    get picked up. Once a span of text has contributed a code, it's
    not matched again by a lower-priority pattern (a bare 6-digit
    pattern shouldn't re-match digits already captured by the more
    specific "hsn code X" pattern).
    """
    found = []
    seen_spans = []

    def overlaps(span):
        return any(s[0] < span[1] and span[0] < s[1] for s in seen_spans)

    for pattern in HSN_CODE_PATTERNS:
        for m in re.finditer(pattern, query, re.IGNORECASE):
            span = m.span(1)
            if overlaps(span):
                continue
            seen_spans.append(span)
            code = re.sub(r'\s+', '', m.group(1))
            if code not in found:
                found.append(code)

    return found


def resolve_hsn_codes_mixed(codes: list):
    """
    Resolves a list of HSN codes the way resolve_hsn_reference() does per
    code, but tracks which ones came up empty and, for those, fetches
    their parent chapter's raw full-text doc as a fallback context source.

    This is the multi-code equivalent of the single-code "hsn_code_via_
    chapter_fallback" path: a code coming up empty in `hsn_codes` usually
    just means that code's chapter hasn't been through hsn_extract_
    pipeline's structured extraction yet, not that the code doesn't
    exist — and with several codes requested together, it's common for
    SOME to be in already-processed chapters (structured, instant,
    template-able) and others not (need the raw chapter table text and
    an LLM pass to extract the right rows). Without this, a 4-code
    request where 2 codes' chapters were unprocessed used to either drop
    those 2 silently or, via a vague follow-up like "make it tabular",
    fall back to a single chapter's full text with no bound on how much
    of that chapter to output — producing exactly the MAX_TOKENS cutoff
    pattern seen on "make it in tabular form" after a 4-code request.

    Returns (resolved_docs, unresolved_codes, chapter_fallback_docs).
    """
    resolved_docs, unresolved = [], []
    for c in codes:
        docs = resolve_hsn_reference(c)
        if docs:
            resolved_docs.extend(docs)
        else:
            unresolved.append(c)

    chapter_docs, seen_chapters = [], set()
    for c in unresolved:
        chapter = c[:2]
        if chapter in seen_chapters:
            continue
        seen_chapters.add(chapter)
        doc = lookup_by_chapter_number(chapter)
        if doc:
            chapter_docs.append(doc)

    return resolved_docs, unresolved, chapter_docs


HSN_MIXED_LOOKUP_GUIDANCE = """
Additional instruction for this specific request: the user asked about
these exact HSN/tariff codes: {codes}.

Some of these codes have structured data provided directly above (use
those exact field values — don't alter or re-derive them). For any
codes WITHOUT structured data, you've been given excerpt(s) from the
chapter's raw tariff text — each code may have MULTIPLE excerpts
(marked with "[...]" between them) because the code can appear in both
the MAIN tariff table AND in embedded notification/exemption tables.

IMPORTANT: prefer the MAIN tariff table entry. The main tariff table
has the real product description (e.g. "Jawar", "Bajra", "Rice,
parboiled") and the base standard rate. Embedded notification tables
(often showing "All Goods" as description and a different BCD rate)
are secondary — mention them only as a note if relevant.

Present ALL of the requested codes together in ONE markdown table with
columns: HSN Code | Description | Unit | Standard Rate | Preferential
Rate. Include exactly the codes requested — no extra rows for other
codes from the same chapter/table that weren't asked about. If a
requested code genuinely doesn't appear anywhere in the context given,
say so in that row rather than guessing.
"""


def resolve_hsn_reference(code: str):
    """
    Resolves a single HSN code reference to a list of matching documents.
    Handles three cases:
      - exact/unambiguous prefix match -> [doc]
      - heading code with multiple children (e.g. "3904 30" / "390430"
        covers both "39043010" and "39043090", neither of which has a
        rate of its own) -> [child_doc, child_doc, ...]
      - no match at all -> []
    Used wherever a single extracted code needs turning into real
    document(s) — direct lookup, comparisons, and follow-up resolution
    all go through this so a heading code is handled the same way
    everywhere instead of silently dropping it.
    """
    doc = lookup_by_hsn_code(code)
    if doc:
        return [doc]
    children = lookup_hsn_children(code)
    return children  # [] if nothing matched at all


def lookup_by_hsn_code(hsn_code: str):
    """
    Exact-match lookup against the 'hsn_codes' collection. Tries an exact
    match first; if that fails, tries treating the query code as a
    prefix (e.g. a 6-digit query "390729" matching an 8-digit stored code
    "39072910"), since users often don't know/type the full 8-digit code.

    A 6-digit code can be either a real standalone tariff line OR a
    "heading" with multiple child sub-codes underneath it (e.g. "3904 30"
    Vinyl chloride-vinyl acetate copolymers has no rate of its own — the
    real rated lines are "3904 30 10" and "3904 30 90", see Chapter 39).
    If the prefix matches MORE THAN ONE document, silently returning just
    one via find_one() would hide the others. In that case this returns
    None and the caller should use lookup_hsn_children() instead to get
    every matching child explicitly — resolve_hsn_reference() above does
    exactly this automatically and is the preferred entry point.

    Returns the matching document, or None if there's no exact match and
    either zero or 2+ prefix matches (ambiguous heading case).
    """
    col = get_hsn_collection()

    doc = col.find_one({"hsn_code": hsn_code}, {"_id": 0})
    if doc:
        return doc

    prefix_pattern = re.compile(f"^{re.escape(hsn_code)}")
    matches = list(col.find({"hsn_code": prefix_pattern}, {"_id": 0}).limit(2))
    if len(matches) == 1:
        return matches[0]
    return None  # 0 matches, or 2+ (ambiguous heading) — see lookup_hsn_children()


def lookup_hsn_children(hsn_code: str, limit: int = 10):
    """
    Returns ALL documents whose hsn_code starts with the given prefix —
    used when a query code turns out to be a heading (e.g. "3904 30")
    rather than a leaf tariff line, so the comparison/answer can show
    every child code instead of silently picking one.
    """
    col = get_hsn_collection()
    prefix_pattern = re.compile(f"^{re.escape(hsn_code)}")
    return list(col.find({"hsn_code": prefix_pattern}, {"_id": 0}).limit(limit))


# ─────────────────────────────────────────────────────────────
# HSN lookup BY NAME — "what's the HSN code for ferro-nickel"
# ─────────────────────────────────────────────────────────────
# Everything above resolves a query that already contains a code. This
# handles the reverse direction: the user names a product/material and
# wants the matching code(s) back. No text index exists on hsn_codes
# (only `notifications` has one), so this uses a plain case-insensitive
# regex against `description` — fine at this collection's size/shape
# (short, mostly-English tariff descriptions), and consistent with the
# project's standing rule of re.compile() over dict-style $regex.
NAME_SEARCH_INTENT_RE = re.compile(
    r'(?:hsn\s*code\s*(?:for|of)?|find\s*(?:the\s*)?hsn\s*code\s*(?:for|of)?|'
    r'what(?:\'?s| is)\s*(?:the\s*)?hsn\s*code\s*(?:for|of)?|'
    r'classification\s*(?:code\s*)?(?:for|of)|tariff\s*code\s*(?:for|of)?)'
    r'\s*[:\-]?\s*(.+)',
    re.IGNORECASE,
)

_NAME_SEARCH_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "is", "what", "code", "hsn",
    "import", "export", "duty", "rate", "item", "items", "product", "products",
}


def looks_like_name_search(query: str) -> bool:
    """True when the question is clearly asking 'what HSN code applies to
    X' rather than already containing a code — i.e. NAME_SEARCH_INTENT_RE
    matches AND there's no code-shaped token already in the query (if
    there is one, the existing code-based lookup should win instead)."""
    return bool(NAME_SEARCH_INTENT_RE.search(query)) and not extract_hsn_codes(query)


def extract_name_search_term(query: str):
    """Pulls the product/material name out of a 'HSN code for X' style
    question. Returns None if the intent pattern doesn't match."""
    m = NAME_SEARCH_INTENT_RE.search(query)
    if not m:
        return None
    term = m.group(1).strip(" ?.!:;,")
    return term or None


def search_hsn_by_name(term: str, limit: int = 15):
    """
    Finds HSN rows whose description matches `term`. Two passes:
      1. Whole-phrase substring match (handles "ferro-nickel", "natural
         rubber latex" etc. matching as-typed).
      2. If that finds nothing, fall back to an OR match on individual
         significant words (3+ letters, stopwords excluded) — e.g.
         "rubber tyres for cars" won't appear verbatim in any single
         description, but rows containing "rubber" or "tyres" will.
         Results are then ranked by how many distinct query words they
         matched, most-relevant first, since plain regex OR has no
         native relevance scoring the way a text index would.
    Returns a list of docs (possibly empty), capped at `limit`.
    """
    col = get_hsn_collection()
    term = term.strip()
    if not term:
        return []

    whole_pattern = re.compile(re.escape(term), re.IGNORECASE)
    docs = list(col.find({"description": whole_pattern}, {"_id": 0}).limit(limit))
    if docs:
        return docs

    words = [w for w in re.findall(r"[A-Za-z\-]{3,}", term) if w.lower() not in _NAME_SEARCH_STOPWORDS]
    if not words:
        return []

    word_patterns = [re.compile(re.escape(w), re.IGNORECASE) for w in words]
    candidates = list(col.find({"$or": [{"description": p} for p in word_patterns]}, {"_id": 0}).limit(limit * 4))

    def relevance(doc):
        desc = doc.get("description", "")
        return sum(1 for p in word_patterns if p.search(desc))

    candidates.sort(key=relevance, reverse=True)
    return candidates[:limit]


def format_hsn_table_answer(docs: list) -> str:
    """
    Deterministic, zero-LLM markdown table for 2+ HSN codes looked up
    together — "give me 2923, 3907, and 4011" or "compare X and Y".
    Same rationale as format_hsn_answer_template(): every field is
    already structured data, so a table is built directly rather than
    spending an LLM call (and risking a misstated rate, or the model
    drifting into prose/bullets instead of the table format that was
    actually asked for) to reformat data that's already in the right shape.
    """
    header = "| HSN Code | Description | Unit | Standard Rate | Preferential Rate | Chapter |"
    divider = "|---|---|---|---|---|---|"
    rows = [header, divider]
    footnotes = []
    for d in docs:
        code = d.get("hsn_code", "N/A")
        desc = (d.get("description") or "(no description on file)").replace("|", "/")
        unit = d.get("unit") or "-"
        std = d.get("standard_rate") or "-"
        pref = d.get("preferential_rate") or "-"
        chapter = d.get("chapter") or "-"
        rows.append(f"| {code} | {desc} | {unit} | {std} | {pref} | {chapter} |")
        if d.get("footnote_text"):
            footnotes.append(f"- **{code}**: {d['footnote_text']}")

    out = "\n".join(rows)
    if footnotes:
        out += "\n\n**Notes:**\n" + "\n".join(footnotes)
    return out


def format_hsn_name_search_answer(term: str, docs: list) -> str:
    """Deterministic, zero-LLM answer listing every HSN row whose
    description matched the searched-for name."""
    if not docs:
        return (
            f"No HSN code matching \"{term}\" was found in the chapters processed so far. "
            f"This could mean the term doesn't appear verbatim in any tariff description, or "
            f"that the relevant chapter hasn't been run through the extraction pipeline yet — "
            f"try a more specific or differently-worded term, or look it up by chapter/code "
            f"directly if you know either one."
        )
    lines = [f"Found {len(docs)} HSN code{'s' if len(docs) != 1 else ''} matching \"{term}\":\n"]
    for d in docs:
        code = d.get("hsn_code", "N/A")
        desc = d.get("description") or "(no description on file)"
        std = d.get("standard_rate") or "not specified"
        chapter = d.get("chapter")
        line = f"- **{code}** — {desc} — Standard Rate: {std}"
        if chapter:
            line += f" (Chapter {chapter})"
        lines.append(line)
    lines.append(
        "\nAsk for any of these codes directly (or ask me to compare a few of them) for full "
        "details including preferential rate and footnotes."
    )
    return "\n".join(lines)


def build_context_block_from_hsn_doc(doc: dict) -> str:
    """Same shape as build_context_block_from_doc(), for a single HSN row.
    Still used as the LLM context block for --mode llm-style answers if
    you ever want Gemini's phrasing back; format_hsn_answer_template()
    below is the default path now (no LLM call)."""
    parts = [f"HSN Code: {doc.get('hsn_code')}", f"Description: {doc.get('description', 'N/A')}"]
    if doc.get("unit"):
        parts.append(f"Unit: {doc['unit']}")
    if doc.get("standard_rate"):
        parts.append(f"Standard Rate of Duty: {doc['standard_rate']}")
    if doc.get("preferential_rate"):
        parts.append(f"Preferential Areas Rate of Duty: {doc['preferential_rate']}")
    if doc.get("footnote_text"):
        parts.append(f"Note: {doc['footnote_text']}")
    parts.append(f"Chapter: {doc.get('chapter', 'N/A')} | Source: {doc.get('source_pdf', 'N/A')}")
    return "=== HSN Code Lookup ===\n" + "\n".join(parts)


def build_context_block_from_hsn_docs(docs: list) -> str:
    """
    Multi-code version of build_context_block_from_hsn_doc(), used when
    a question references 2+ HSN codes (e.g. "compare 2923 20 90 with
    2933 39 90", "import duty difference between X and Y"). Concatenates
    each code's structured data into its own labeled block so the LLM
    has every code's full record in context at once, instead of only
    the first code (the templated zero-LLM path only ever handles a
    single code and would otherwise silently drop the rest).
    """
    return "\n\n".join(build_context_block_from_hsn_doc(d) for d in docs)


def format_hsn_answer_template(doc: dict) -> str:
    """
    Deterministic, zero-LLM answer for a direct HSN code match. Every field
    used here is already structured data straight from MongoDB (no prose
    to summarize, nothing ambiguous to phrase) — having Gemini "rewrite"
    a duty rate into a sentence spends a call to do something a template
    does for free, with zero risk of the rate getting misstated in the
    rephrasing. Falls back to the LLM context-block path (build_context_
    block_from_hsn_doc) only if you explicitly want LLM phrasing instead.
    """
    code = doc.get("hsn_code", "N/A")
    desc = doc.get("description") or "(no description on file)"
    unit = doc.get("unit")
    std = doc.get("standard_rate")
    pref = doc.get("preferential_rate")
    footnote = doc.get("footnote_text")

    lines = [f"**HSN Code {code}** — {desc}"]
    if unit:
        lines.append(f"Unit: {unit}")
    if std:
        lines.append(f"Standard Rate of Duty: {std}")
    else:
        lines.append("Standard Rate of Duty: not specified in this record")
    if pref:
        lines.append(f"Preferential Areas Rate of Duty: {pref}")
    if footnote:
        lines.append(f"Note: {footnote}")
    chapter = doc.get("chapter")
    if chapter:
        lines.append(f"(Tariff Schedule, Chapter {chapter})")

    return "\n".join(lines)


def format_hsn_heading_answer(queried_code: str, children: list) -> str:
    """
    Deterministic, zero-LLM answer for when the queried code is a
    "heading" — it has no tariff line/rate of its own, but 2+ child
    codes underneath it do (e.g. querying "3904 30" should surface both
    "3904 30 10" and "3904 30 90" rather than silently picking one via
    find_one(), which is what lookup_by_hsn_code() used to do before
    resolve_hsn_reference() made this case explicit).
    """
    lines = [f"**HSN {queried_code}** has no rate of its own — it's a heading with {len(children)} sub-codes:"]
    for d in children:
        code = d.get("hsn_code", "N/A")
        desc = d.get("description") or "(no description on file)"
        std = d.get("standard_rate") or "not specified"
        lines.append(f"\n**HSN Code {code}** — {desc}")
        lines.append(f"Standard Rate of Duty: {std}")
        if d.get("preferential_rate"):
            lines.append(f"Preferential Areas Rate of Duty: {d['preferential_rate']}")
    chapter = children[0].get("chapter") if children else None
    if chapter:
        lines.append(f"\n(Tariff Schedule, Chapter {chapter})")
    return "\n".join(lines)


def retrieve_context(query: str, top_k: int = 5, category=None) -> list:
    """
    Returns the top_k most relevant chunks for a query as a list of dicts
    (text, notification_no, title, category, date, pdf_url, score).

    `category` can be:
      - a plain string, e.g. "GST"            -> exact match filter
      - a dict, e.g. {"$ne": "Tariff Schedule"} -> passed through as-is
      - None                                   -> no category filter
    """
    col = get_chunks_collection()
    query_vector = embed_query(query)

    vector_stage = {
        "$vectorSearch": {
            "index": VECTOR_INDEX,
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": max(100, top_k * 20),
            "limit": top_k,
        }
    }
    if category:
        vector_stage["$vectorSearch"]["filter"] = {"category": category}

    pipeline = [
        vector_stage,
        {"$project": {
            "_id": 0, "text": 1, "notification_id": 1, "title": 1,
            "category": 1, "notification_no": 1, "date": 1, "pdf_url": 1,
            "score": {"$meta": "vectorSearchScore"},
        }},
    ]
    return list(col.aggregate(pipeline))


# ─────────────────────────────────────────────────────────────
# Category auto-filtering
# ─────────────────────────────────────────────────────────────
# The 'chunks' collection mixes two very different kinds of documents
# under one schema (see project.py's is_tariff_schedule_doc()):
#   - "Tariff Schedule": static HSN chapter listings (chap-1.pdf, etc.)
#     with no notification number, date, or policy content
#   - everything else: actual regulatory notifications, circulars, etc.
# Mixing them in vector search hurts precision for notification-style
# questions, since chapter listings can rank highly on pure semantic
# similarity (e.g. a question about "polymer import duty" can match a
# tariff chapter just because the word "polymer" appears in a heading).
#
# Default behavior: exclude "Tariff Schedule" unless the question is
# itself clearly about a tariff/chapter lookup. Callers that pass an
# explicit `category` always take precedence over this auto-filter.
TARIFF_SCHEDULE_CATEGORY = "Tariff Schedule"

_CHAPTER_INTENT_PATTERN = re.compile(
    r'\b(chapter|chap)\s*\.?\s*\d+\b|\btariff\s+schedule\b|\bhsn\b', re.IGNORECASE
)


def looks_like_chapter_query(query: str) -> bool:
    """True if the question is itself about a tariff chapter / HSN listing
    rather than an actual notification, e.g. 'what's in chapter 39'.
    Also true for a bare HSN-code-shaped query like "7202 60 00 import
    duty" — an HSN code is unambiguously Tariff Schedule territory even
    without the word "chapter" appearing, and excluding Tariff Schedule
    docs from vector search for these was producing false "documents
    don't contain this" answers for codes that simply hadn't been
    through hsn_extract_pipeline's structured extraction yet (the
    chapter-level fallback in chat_with_context now catches most of
    these before reaching vector search at all — this is the remaining
    safety net for cases that fall through anyway)."""
    if _CHAPTER_INTENT_PATTERN.search(query):
        return True
    return bool(extract_hsn_codes(query))


def resolve_category_filter(query: str, requested_category: str = None):
    """
    Decides which category filter (if any) to apply to vector search.
      - Explicit category param always wins (manual override).
      - Otherwise, if the question looks like a chapter/tariff-schedule
        query, don't filter (let it search Tariff Schedule docs).
      - Otherwise, default to excluding Tariff Schedule, since the vast
        majority of real questions are about notifications/policy, not
        HSN chapter listings.
    Returns either a category string (positive filter) or a dict in
    Mongo's $vectorSearch filter shape (e.g. exclusion via $ne), or None.
    """
    if requested_category:
        return requested_category
    if looks_like_chapter_query(query):
        return None
    return {"$ne": TARIFF_SCHEDULE_CATEGORY}


SYSTEM_PROMPT_TEMPLATE = """You are RegulAI, an expert AI assistant specialized in Indian trade law, regulatory compliance, customs duties, import/export regulations, and government notifications.

You are analyzing the following regulatory notifications, retrieved via semantic search for relevance to the user's question:

--- REGULATORY DOCUMENTS ---
{context}
--- END DOCUMENTS ---

Your role:
- Answer questions about regulatory notifications, trade laws, customs duties, import/export restrictions.
- Always cite the specific Notification Number when referencing content, e.g. "As per Notification No. 12/2023-Customs..."
- Identify amendments, superseded notifications, and regulatory relationships.
- Highlight compliance requirements clearly.
- When listing regulations, use structured formatting with bullet points.
- Flag any prohibitions, restrictions, or exemptions clearly.
- If asked to compare notifications, do so in a structured table format.
- If the documents above don't contain the answer, state that clearly instead of guessing.
- If the user states a fact, code, description, or rate themselves (rather than it appearing in the documents above), do not repeat it back as if it were confirmed by the documents. Acknowledge it as something they provided and say plainly whether or not you can verify it against the retrieved documents.

Format your responses professionally for compliance teams and legal departments.
Use **bold** for notification numbers, key terms, and important compliance points.
"""

# Appended only for chapter-vs-chapter comparisons (not HSN-code comparisons,
# which are already small/structured and don't need bounding). Two full
# tariff chapters can be up to 36,000 input characters combined, and without
# an explicit bound the model tends to try to be exhaustive over all of it —
# which is what was actually blowing past a 4000-token output budget, not a
# lack of budget on its own. Bounding the task is more robust than just
# raising max_tokens again, since chapter text length varies a lot and will
# only grow as more chapters get added to the pipeline.
CHAPTER_COMPARISON_GUIDANCE = """
Additional instruction for this comparison specifically: the two chapters
above are long. Do NOT attempt to transcribe or exhaustively cover every
clause. Instead:
- Do NOT use a markdown table for this comparison. Use a bulleted
  side-by-side format instead: one bullet per comparison point, each
  bullet giving both chapters' value for that point in a single line,
  e.g. "- **Scope:** Chapter X covers ... ; Chapter Y covers ...".
  (Wide multi-column markdown tables for two full-chapter comparisons
  have repeatedly caused this model to stall or loop mid-table — stick
  to bullets even if a table feels like the more natural format.)
- Limit this to at most 8-10 bullets, covering only the most materially
  important differences (scope, key exclusions, general classification
  approach, notable rate patterns).
- This bound applies even if the user specifically asks to "compare duty
  rates" — a full chapter can have 50-900+ individual tariff lines, each
  with its own rate, and enumerating them all is exactly the failure mode
  this guidance exists to prevent. Summarize the rate PATTERN instead
  (e.g. "most headings sit at 10%, with a few exceptions at 25% for X and
  2.5% for Y") and offer to look up the exact rate for any specific HSN
  code the user names, rather than listing every code's rate.
- Follow the bullets with a short paragraph (3-5 sentences max) of analysis.
- If there's more detail the user might want, say so explicitly and invite
  a follow-up question about a specific aspect, rather than including it.
Keep the entire response well under 3000 words.
"""

MAX_HISTORY_TURNS = 6


def trim_history(messages: list) -> list:
    if len(messages) <= MAX_HISTORY_TURNS * 2:
        return messages
    return messages[-(MAX_HISTORY_TURNS * 2):]


def build_context_block(chunks: list) -> str:
    if not chunks:
        return "No relevant CBIC documents were found in the database for this query."
    blocks = []
    for c in chunks:
        ref = c.get("notification_no") or c.get("notification_id")
        blocks.append(
            f"=== Notification: {ref} | {c.get('title', '')[:80]} | "
            f"Date: {c.get('date', 'N/A')} | Category: {c.get('category', 'N/A')} ===\n{c.get('text', '')}"
        )
    return "\n\n".join(blocks)


def extract_lines_for_hsn_codes(full_text: str, codes: list, context_chars: int = 400) -> str:
    """
    Given a chapter's raw full_text, finds EVERY occurrence of each
    requested HSN code and returns snippets around all of them.

    Why all occurrences: a chapter's full_text commonly contains the
    code twice — once in the main tariff table (with the real description
    like "Jawar" / "50%") and again in an embedded notification/exemption
    table (with generic "All Goods" and a different BCD rate). Grabbing
    only the first occurrence lands on whichever the PDF happened to put
    first, which is often the notification table — producing "All Goods /
    30" when the user wanted "Jawar / 50%". Returning all snippets lets
    the LLM see both and pick the main tariff table entry.
    """
    blocks = []
    for code in codes:
        spaced = code
        compact = code.replace(" ", "")
        snippets = []
        pos = 0
        while len(snippets) < 4:   # cap: 4 occurrences per code is plenty
            idx = full_text.find(spaced, pos)
            if idx == -1:
                idx = full_text.find(compact, pos)
                if idx == -1:
                    break
            start = max(0, idx - 60)
            end = min(len(full_text), idx + context_chars)
            snippets.append(full_text[start:end].strip())
            pos = idx + max(len(spaced), len(compact))

        if not snippets:
            blocks.append(f"[Code {code}: not found in chapter text]")
        else:
            joined = "\n\n[...]\n\n".join(snippets)
            blocks.append(f"--- Code {code} ({len(snippets)} occurrence(s)) ---\n{joined}")
    return "\n\n".join(blocks)


def build_context_block_from_doc(doc: dict, max_chars: int = None) -> str:
    """
    Same shape as build_context_block(), but for a single full document
    returned by the direct notification-number lookup rather than a list
    of vector-search chunks. Uses full_text since we have the whole
    document, not just a chunk of it.

    max_chars: if given, truncates full_text to this many characters
    (with a note appended) before building the block. Used by the
    multi-doc comparison path below — feeding two full chapters'
    full_text (up to 50,000 chars each, per MAX_TEXT_CHARS in
    mongo_pipeline.py) into a single prompt is ~25k+ tokens of input
    before the system prompt/instructions are even added, which made
    truncated/cut-off comparison responses far more likely. A single-
    document lookup keeps using the untruncated full_text (max_chars=None).
    """
    ref = doc.get("notification_no") or doc.get("notification_id")
    text = doc.get("full_text") or "(No extracted text available for this document.)"
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated for length — full chapter text is longer than shown here...]"
    return (
        f"=== Notification: {ref} | {doc.get('title', '')[:80]} | "
        f"Date: {doc.get('date', 'N/A')} | Category: {doc.get('category', 'N/A')} ===\n{text}"
    )


def build_context_block_from_docs(docs: list, max_chars_each: int = 18_000) -> str:
    """
    Multi-document version of build_context_block_from_doc(), used when
    a question references 2+ chapters or notifications at once (e.g.
    "compare chapter 29 and chapter 39"). Without this, only the first
    chapter's full_text would ever reach the LLM, so a comparison would
    either talk about one chapter only or hallucinate the other from
    training knowledge — there'd be nothing about the second chapter
    in {context} at all.

    Each document is capped at max_chars_each (default 18,000) rather
    than using the full up-to-50,000-char full_text per chapter. For a
    comparison (scope/exclusions/classification rules/rates), the chapter
    notes at the front of full_text carry the content that actually
    matters — the back end of a long chapter is mostly repetitive tariff
    line items. Uncapped, 2 chapters could mean ~100k chars (~25k+
    tokens) of input on top of the system prompt, which was producing
    truncated/cut-off comparison responses in practice.
    """
    return "\n\n".join(build_context_block_from_doc(d, max_chars=max_chars_each) for d in docs)


def chat_with_context(messages: list, top_k: int = 5, category: str = None) -> dict:
    """
    Multi-turn RAG chat. `messages` is the full conversation so far:
    [{"role": "user"/"assistant", "content": ...}, ...].

    Retrieval strategy for the latest user question:
      1. Direct lookup — if the question contains something that looks
         like a notification number (e.g. "50/2017") or a tariff chapter
         reference (e.g. "chapter 2"), try an exact match against the
         'notifications' collection first. This bypasses vector search
         entirely: a bare numeric code has no semantic content for an
         embedding model to latch onto (chapter 2 isn't "closer in
         meaning" to chapter 4 than to chapter 84), so vector search
         alone returns near-arbitrary results for these queries.
      2. Vector search fallback — if no notification-number- or
         chapter-shaped token is found in the question, or the lookup
         finds nothing for the token that was found, fall back to the
         existing $vectorSearch flow over 'chunks'.

    Returns {response, sources, retrieval_method}.
    """
    latest_question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    # ── Name-based HSN search ("HSN code for ferro-nickel") ─────────
    # Checked first and separately from everything below: this answers a
    # fundamentally different question (name -> code) than the rest of
    # this function (code/chapter/notification-number -> details), and a
    # query like "hsn code for natural rubber" would otherwise fall
    # through every other branch and hit generic vector search, which
    # has no path back to a specific hsn_codes row. Templated/zero-LLM,
    # same rationale as the rest of the structured HSN answers.
    if looks_like_name_search(latest_question):
        term = extract_name_search_term(latest_question)
        if term:
            matches = search_hsn_by_name(term)
            return {
                "response": format_hsn_name_search_answer(term, matches),
                "sources": [{
                    "notification_no": None,
                    "title": f"HSN {d.get('hsn_code')} — {d.get('description', '')[:80]}",
                    "category": "HSN Code",
                    "pdf_url": None,
                    "score": None,
                } for d in matches],
                "retrieval_method": "name_search:hsn_code (explicit intent, no LLM call)",
            }

    direct_doc = None
    direct_lookup_kind = None
    is_hsn_doc = False
    multi_docs = None          # list of docs when 2+ entities of the same kind are referenced
    multi_kind = None          # "hsn_code" / "hsn_code_mixed" / "chapter_number"
    mixed_requested_codes = None   # set alongside multi_kind == "hsn_code_mixed"
    mixed_chapter_docs = None      # raw chapter docs backing the unresolved codes

    # ── Multi-entity check FIRST ──────────────────────────────────
    # A comparison question ("compare HSN 2923 20 90 with 2933 39 90",
    # "compare chapter 29 and chapter 39") contains 2+ codes/chapters of
    # the SAME kind. This must be checked before the single-entity path
    # below: extract_hsn_code()/extract_chapter_number() only ever look
    # at the first match, so without this check a 2-code question would
    # silently resolve as if only the first code had been mentioned —
    # which was the original bug (instant single-code answer, rest of
    # the prompt ignored; chapter comparisons only ever seeing one
    # chapter's text and hallucinating or ignoring the other).
    hsn_codes = extract_hsn_codes(latest_question)
    if len(hsn_codes) >= 2:
        docs, unresolved, chapter_docs = resolve_hsn_codes_mixed(hsn_codes)
        if unresolved and (docs or chapter_docs):
            # Some codes resolved cleanly, some need chapter-text fallback
            # (or none resolved at all but their chapters do exist) — mixed
            # path, one LLM call to assemble a single table across both
            # sources rather than a deterministic table missing rows.
            multi_docs, multi_kind = docs, "hsn_code_mixed"
            mixed_requested_codes, mixed_chapter_docs = hsn_codes, chapter_docs
        elif len(docs) >= 2:
            multi_docs, multi_kind = docs, "hsn_code"

    if multi_docs is None:
        chapter_nos = extract_chapter_numbers(latest_question)
        if len(chapter_nos) >= 2:
            docs = [d for d in (lookup_by_chapter_number(c) for c in chapter_nos) if d]
            if len(docs) >= 2:
                multi_docs, multi_kind = docs, "chapter_number"

    # ── Multi-entity follow-up resolution ───────────────────────────
    # Catches "compare duty rate of both the chapters", "make it in
    # tabular form", "format that as a table", "what about each of
    # them" — any vague follow-up referencing a previously-established
    # set of 2+ entities, with no explicit numbers/codes of its own.
    # Without this, such a question fell through to the single-entity
    # follow-up path below, which only ever recovers ONE entity —
    # degrading e.g. a 4-HSN-code request into a single-code (or
    # single-chapter-fallback) answer, with no bound on output, which is
    # exactly what was producing the MAX_TOKENS cutoffs on "make it in
    # tabular form" after a multi-code question. Gated on the same
    # looks_like_vague_followup() check as the single-entity path, not
    # on specific wording — see find_recent_multi_entity_followup()
    # docstring for why wording-based detection doesn't generalize here.
    if multi_docs is not None and looks_like_vague_followup(latest_question):
        kind, values = find_recent_multi_entity_followup(messages)
        if kind == "hsn_code" and len(values) >= 2:
            resolved, unresolved, chapter_docs = resolve_hsn_codes_mixed(values)
            if unresolved and (resolved or chapter_docs):
                multi_docs = resolved
                multi_kind = "hsn_code_mixed"
                mixed_requested_codes, mixed_chapter_docs = values, chapter_docs
            elif len(resolved) >= 2:
                multi_docs, multi_kind = resolved, "hsn_code"
        elif kind == "chapter_number" and len(values) >= 2:
            docs = [d for d in (lookup_by_chapter_number(c) for c in values) if d]
            if len(docs) >= 2:
                multi_docs, multi_kind = docs, "chapter_number"

    if multi_docs is None and looks_like_vague_followup(latest_question):
        kind, values = find_recent_multi_entity_followup(messages)
        if kind == "hsn_code" and len(values) >= 2:
            resolved, unresolved, chapter_docs = resolve_hsn_codes_mixed(values)
            if unresolved and (resolved or chapter_docs):
                multi_docs = resolved
                multi_kind = "hsn_code_mixed"
                mixed_requested_codes, mixed_chapter_docs = values, chapter_docs
            elif len(resolved) >= 2:
                multi_docs, multi_kind = resolved, "hsn_code"
        elif kind == "chapter_number" and len(values) >= 2:
            docs = [d for d in (lookup_by_chapter_number(c) for c in values) if d]
            if len(docs) >= 2:
                multi_docs, multi_kind = docs, "chapter_number"

    if multi_docs is not None:
        if multi_kind == "hsn_code":
            # Templated, zero-LLM: this is exactly "give me these codes in
            # a table", which is fully satisfied by structured data alone
            # — no synthesis needed, and a deterministic table guarantees
            # the format actually asked for instead of hoping the model
            # chooses a table over bullets/prose.
            return {
                "response": format_hsn_table_answer(multi_docs),
                "sources": [{
                    "notification_no": None,
                    "title": f"HSN {d.get('hsn_code')} — {d.get('description', '')[:80]}",
                    "category": "HSN Code",
                    "pdf_url": None,
                    "score": None,
                } for d in multi_docs],
                "retrieval_method": f"direct_lookup:hsn_code_multi ({len(multi_docs)} codes, templated, no LLM call)",
            }

        if multi_kind == "hsn_code_mixed":
            # Mixed path: some codes resolved to structured rows, others
            # only have their parent chapter's raw tariff table text.
            # IMPORTANT: don't pass the whole chapter as context — a full
            # chapter is ~50k chars and at max_tokens=800 the system prompt
            # alone leaves almost no room for output, which is exactly what
            # was producing MAX_TOKENS cutoffs on 3-row tables. Instead,
            # extract just the ~400 chars around each unresolved code from
            # the chapter's full_text (extract_lines_for_hsn_codes), so the
            # total context for a 4-code request stays under ~2k chars.
            structured_block = build_context_block_from_hsn_docs(multi_docs) if multi_docs else ""

            unresolved_by_chapter = {}
            for code in mixed_requested_codes:
                if not any(d.get("hsn_code", "").replace(" ", "") == code.replace(" ", "") for d in multi_docs):
                    chapter = code.replace(" ", "")[:2]
                    unresolved_by_chapter.setdefault(chapter, []).append(code)

            snippet_blocks = []
            for chap_doc in (mixed_chapter_docs or []):
                chap_no = chap_doc.get("notification_id", "").replace("chap-", "")
                codes_for_this_chapter = unresolved_by_chapter.get(chap_no, [])
                if not codes_for_this_chapter:
                    # try two-digit match on any key
                    for k, v in unresolved_by_chapter.items():
                        if chap_no.lstrip("0") == k.lstrip("0"):
                            codes_for_this_chapter = v
                            break
                full_text = chap_doc.get("full_text", "")
                if full_text and codes_for_this_chapter:
                    snippet_blocks.append(extract_lines_for_hsn_codes(full_text, codes_for_this_chapter))

            chapter_snippet = "\n\n".join(snippet_blocks)
            context = "\n\n".join(filter(None, [structured_block, chapter_snippet]))

            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
            system_prompt += HSN_MIXED_LOOKUP_GUIDANCE.format(
                codes=", ".join(mixed_requested_codes)
            )

            sources = (
                [{
                    "notification_no": None,
                    "title": f"HSN {d.get('hsn_code')} — {d.get('description', '')[:80]}",
                    "category": "HSN Code",
                    "pdf_url": None,
                    "score": None,
                } for d in multi_docs]
                + [{
                    "notification_no": d.get("notification_no"),
                    "title": d.get("title"),
                    "category": d.get("category"),
                    "pdf_url": d.get("pdf_url"),
                    "score": None,
                } for d in (mixed_chapter_docs or [])]
            )

            llm_messages = [
                {"role": ("user" if msg["role"] == "user" else "assistant"), "content": msg["content"]}
                for msg in trim_history(messages)
            ]

            result = call_gemini(
                messages=llm_messages,
                system_prompt=system_prompt,
                # Small budget: a table of N rows is ~50-80 tokens per
                # row, so 800 covers up to ~10 codes with room to spare.
                # Thinking off — this is rote extraction with no
                # ambiguity to reason about, and thinking_budget=0
                # is safe here because the output is short/bounded
                # (the degenerate-repetition failure only appeared on
                # unbounded, open-ended synthesis tasks).
                max_tokens=800,
                thinking_budget=0,
            )
            print(f"🔍 Gemini usage (hsn_code_mixed, {len(mixed_requested_codes)} codes): {result['usage']}")
            response_text = result["text"]
            finish_reason = result["finish_reason"]
            if finish_reason and finish_reason not in ("stop", "STOP"):
                print(f"⚠ Mixed HSN response cut short — finish_reason={finish_reason}")
                response_text += (
                    f"\n\n*(Note: response cut short — finish_reason: {finish_reason}.)*"
                )
            return {
                "response": response_text,
                "sources": sources,
                "retrieval_method": f"direct_lookup:hsn_code_mixed ({len(mixed_requested_codes)} codes, {len(multi_docs)} structured + {len(mixed_chapter_docs or [])} chapter fallback)",
            }

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=build_context_block_from_docs(multi_docs))
        system_prompt += CHAPTER_COMPARISON_GUIDANCE
        sources = [{
            "notification_no": d.get("notification_no"),
            "title": d.get("title"),
            "category": d.get("category"),
            "pdf_url": d.get("pdf_url"),
            "score": None,
        } for d in multi_docs]

        llm_messages = [
            {"role": ("user" if msg["role"] == "user" else "assistant"), "content": msg["content"]}
            for msg in trim_history(messages)
        ]

        result = call_gemini(
            messages=llm_messages,
            system_prompt=system_prompt,
            # CONFIRMED (via Google's own python-genai issue tracker, e.g.
            # googleapis/python-genai#782 and #811): on gemini-2.5-flash,
            # thoughts_token_count counts AGAINST max_output_tokens even
            # though it's invisible, and dynamic thinking (thinking_budget
            # = -1) is explicitly flagged as unreliable in combination with
            # max_output_tokens — the model's own thinking allocation can
            # vary run to run and isn't properly reconciled against the cap,
            # which is exactly the inconsistent MAX_TOKENS-at-different-small-
            # totals behavior seen here (cut at ~70 tokens one run, ~1081 the
            # next). The documented workaround is an EXPLICIT capped thinking
            # budget plus generous headroom above it, not -1. 1024 thinking +
            # plenty of room for the actual bounded answer (which
            # CHAPTER_COMPARISON_GUIDANCE caps at ~3000 words / 8-10 rows).
            max_tokens=10000,
            thinking_budget=1536,
        )
        print(f"🔍 Gemini usage (multi-entity, {multi_kind}): {result['usage']}")
        response_text = result["text"]
        finish_reason = result["finish_reason"]

        # If Gemini stopped because it ran out of output tokens (or any
        # reason other than a clean "stop"), say so explicitly instead of
        # silently handing back a sentence that trails off mid-word. This
        # was previously invisible — the cut-off response was returned
        # as if it were a complete answer, with nothing in the JSON or
        # logs indicating truncation.
        if finish_reason and finish_reason not in ("stop", "STOP"):
            print(f"⚠ Gemini comparison response did not finish cleanly — finish_reason={finish_reason}")
            response_text += (
                f"\n\n*(Note: this response was cut short by the model — finish_reason: "
                f"{finish_reason}. Try asking for a shorter comparison, or ask about one "
                f"chapter at a time.)*"
            )

        return {
            "response": response_text,
            "sources": sources,
            "retrieval_method": f"direct_lookup:{multi_kind}_comparison ({len(multi_docs)} docs)",
        }

    # ── Single-entity path (unchanged behavior) ───────────────────
    notif_no = extract_notification_number(latest_question)
    if notif_no:
        direct_doc = lookup_by_notification_number(notif_no)
        if direct_doc:
            direct_lookup_kind = "notification_number"

    if not direct_doc:
        hsn_code = extract_hsn_code(latest_question)
        if hsn_code:
            hsn_matches = resolve_hsn_reference(hsn_code)
            if len(hsn_matches) == 1:
                direct_doc = hsn_matches[0]
                direct_lookup_kind = "hsn_code"
                is_hsn_doc = True
            elif len(hsn_matches) > 1:
                # Heading code with multiple children (e.g. "3904 30" has
                # no rate of its own — "39043010" and "39043090" do).
                # Answer with all of them rather than silently picking one.
                return {
                    "response": format_hsn_heading_answer(hsn_code, hsn_matches),
                    "sources": [{
                        "notification_no": None,
                        "title": f"HSN {d.get('hsn_code')} — {d.get('description', '')[:80]}",
                        "category": "HSN Code",
                        "pdf_url": None,
                        "score": None,
                    } for d in hsn_matches],
                    "retrieval_method": "direct_lookup:hsn_code_heading (templated, no LLM call)",
                }
            else:
                # CONFIRMED BUG FIX: zero matches here doesn't mean the
                # code doesn't exist — it usually just means that code's
                # chapter hasn't been run through hsn_extract_pipeline's
                # structured extraction yet (per the project's rollout
                # status, only a handful of chapters had been processed
                # into the 'hsn_codes' collection at any given point,
                # with ~90 more pending). The 'notifications' collection,
                # by contrast, has full chapter text for nearly every
                # chapter already (scraped+extracted independently of the
                # HSN pipeline). Previously this case fell straight
                # through to vector search, which (a) has no real signal
                # for a bare numeric code and (b) by default excludes
                # "Tariff Schedule" docs entirely (see
                # resolve_category_filter), so the chapter's real tariff
                # table was never even considered — producing a false
                # "this document doesn't contain that code" answer for
                # data that was actually sitting right there in the
                # chapter's full text. Derive the chapter from the code's
                # first two digits and try that chapter doc directly
                # before giving up on a direct lookup.
                derived_chapter = hsn_code[:2]
                chapter_doc = lookup_by_chapter_number(derived_chapter)
                if chapter_doc:
                    direct_doc = chapter_doc
                    direct_lookup_kind = "hsn_code_via_chapter_fallback"

    if not direct_doc:
        chapter_no = extract_chapter_number(latest_question)
        if chapter_no:
            direct_doc = lookup_by_chapter_number(chapter_no)
            if direct_doc:
                direct_lookup_kind = "chapter_number"

    # Generalized conversational follow-up: if nothing in the current
    # message resolved to a direct lookup, and the message looks like a
    # vague/short follow-up rather than a fresh question, reuse the most
    # recent HSN code / notification number / chapter number from earlier
    # in this conversation. See find_recent_entity_followup() above for
    # why this matters — this is the fix for the "what is description"
    # after an HSN-code lookup falling through to vector search.
    if not direct_doc and looks_like_vague_followup(latest_question):
        kind, value = find_recent_entity_followup(messages)
        if kind == "hsn_code":
            hsn_matches = resolve_hsn_reference(value)
            if len(hsn_matches) == 1:
                direct_doc = hsn_matches[0]
                direct_lookup_kind = "hsn_code_followup"
                is_hsn_doc = True
            elif len(hsn_matches) > 1:
                return {
                    "response": format_hsn_heading_answer(value, hsn_matches),
                    "sources": [{
                        "notification_no": None,
                        "title": f"HSN {d.get('hsn_code')} — {d.get('description', '')[:80]}",
                        "category": "HSN Code",
                        "pdf_url": None,
                        "score": None,
                    } for d in hsn_matches],
                    "retrieval_method": "direct_lookup:hsn_code_heading_followup (templated, no LLM call)",
                }
        elif kind == "notification_number":
            direct_doc = lookup_by_notification_number(value)
            if direct_doc:
                direct_lookup_kind = "notification_number_followup"
        elif kind == "chapter_number":
            direct_doc = lookup_by_chapter_number(value)
            if direct_doc:
                direct_lookup_kind = "chapter_number_followup"

    if direct_doc and is_hsn_doc:
        # Fully structured data - no prose to summarize, nothing for an
        # LLM to add. Answer directly from the template, zero LLM calls.
        return {
            "response": format_hsn_answer_template(direct_doc),
            "sources": [{
                "notification_no": None,
                "title": f"HSN {direct_doc.get('hsn_code')} — {direct_doc.get('description', '')[:80]}",
                "category": "HSN Code",
                "pdf_url": None,
                "score": None,
            }],
            "retrieval_method": "direct_lookup:hsn_code (templated, no LLM call)",
        }

    if direct_doc:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=build_context_block_from_doc(direct_doc))
        if direct_lookup_kind == "hsn_code_via_chapter_fallback":
            # This wasn't a structured single-row hsn_codes match — it's
            # the FULL chapter's raw tariff table text. Point the model
            # at exactly which code the user asked about so it searches
            # the table for that specific line rather than summarizing
            # the chapter in general.
            system_prompt += (
                f"\n\nThe user specifically asked about HSN/tariff code {hsn_code}. "
                f"Find that exact code's row in the tariff table above (description, "
                f"unit, standard rate, preferential rate) and answer with those exact "
                f"values. If that precise code genuinely isn't a row in the table above "
                f"(check both the exact code and its parent heading), say so plainly — "
                f"don't guess at a rate."
            )
        sources = [{
            "notification_no": direct_doc.get("notification_no"),
            "title": direct_doc.get("title"),
            "category": direct_doc.get("category"),
            "pdf_url": direct_doc.get("pdf_url"),
            "score": None,  # exact match, not a similarity score
        }]
        retrieval_method = f"direct_lookup:{direct_lookup_kind}"
    else:
        effective_category = resolve_category_filter(latest_question, category)

        # ── Bare-name HSN fallback (pre-vector-search) ────────────────
        # Catches "bajra", "ferro-nickel", "natural rubber latex" typed
        # without any "HSN code for" phrasing — i.e. looks_like_name_search()
        # returned False above (no explicit intent keywords) but the query
        # also has no codes, no chapter, no notification number, and
        # hasn't matched anything above. Before spending a vector search
        # call (which returns completely wrong docs for product-name queries
        # — "bajra" hit "Project imports" in testing), try the name search
        # against the structured hsn_codes description field. If it finds
        # anything useful, return that directly; if it finds nothing, fall
        # through to vector search as normal so genuinely policy/regulatory
        # queries aren't blocked by this check.
        # Gated: only fires when no other routing succeeded (else branch),
        # the query has no codes/chapters/notifications in it, and the
        # query isn't a conversational follow-up (follow-ups should use
        # the prior context, not start a fresh name search).
        _has_code_or_chapter = bool(extract_hsn_codes(latest_question) or
                                    extract_chapter_numbers(latest_question) or
                                    extract_notification_number(latest_question))
        if not _has_code_or_chapter and not looks_like_vague_followup(latest_question):
            # Strip common request verbiage ("give me information on X",
            # "tell me about X", "details for X") so we search for just
            # the product name, not the whole sentence. Without this,
            # "give me information on bajra" passed as the search term
            # and the formatter printed 'No HSN code matching "give me
            # information on bajra"' instead of "bajra".
            _clean_term = re.sub(
                r'^(?:give\s+me\s+(?:information|details|info)\s+(?:on|about|for)\s+|'
                r'(?:information|details|info)\s+(?:on|about|for)\s+|'
                r'(?:tell\s+me|show\s+me|find|search\s+for|look\s+up)\s+(?:about\s+)?|'
                r'what\s+(?:is|are)\s+(?:the\s+)?)',
                '', latest_question.strip(), flags=re.IGNORECASE
            ).strip(" ?.!:;,")

            _bare_matches = search_hsn_by_name(_clean_term) if _clean_term else []
            if _bare_matches:
                return {
                    "response": format_hsn_name_search_answer(_clean_term, _bare_matches),
                    "sources": [{
                        "notification_no": None,
                        "title": f"HSN {d.get('hsn_code')} — {d.get('description', '')[:80]}",
                        "category": "HSN Code",
                        "pdf_url": None,
                        "score": None,
                    } for d in _bare_matches],
                    "retrieval_method": "name_search:hsn_code (bare-name fallback, no LLM call)",
                }

            # Nothing in the structured hsn_codes collection — may just
            # mean that product's chapter isn't processed yet. Try a
            # vector search scoped to Tariff Schedule and ask Gemini to
            # extract the HSN code from whatever chapter text comes back.
            # Better than dropping to unscoped vector search which
            # returns completely unrelated notification docs for product
            # name queries ("bajra" → "Project imports" in testing).
            if _clean_term and len(_clean_term) >= 3:
                _ts_chunks = retrieve_context(
                    _clean_term, top_k=3, category="Tariff Schedule"
                )
                if _ts_chunks:
                    _ts_context = build_context_block(_ts_chunks)
                    _ts_system = SYSTEM_PROMPT_TEMPLATE.format(context=_ts_context)
                    _ts_system += (
                        f"\n\nThe user is looking for the HSN/tariff code and duty rate "
                        f"for \"{_clean_term}\". Search the tariff table text above for "
                        f"any entry whose description matches or is the common name for "
                        f"\"{_clean_term}\". If found, state the HSN code, full description, "
                        f"unit, standard rate, and preferential rate. If not found in the "
                        f"text above, say so plainly and suggest which chapter they might "
                        f"look in (e.g. Chapter 10 for cereals/grains, Chapter 12 for oil "
                        f"seeds, Chapter 29 for organic chemicals)."
                    )
                    _ts_sources = [{
                        "notification_no": c.get("notification_no"),
                        "title": c.get("title"),
                        "category": c.get("category"),
                        "pdf_url": c.get("pdf_url"),
                        "score": c.get("score"),
                    } for c in _ts_chunks]
                    _ts_messages = [
                        {"role": ("user" if m["role"] == "user" else "assistant"), "content": m["content"]}
                        for m in trim_history(messages)
                    ]
                    _ts_result = call_gemini(
                        messages=_ts_messages,
                        system_prompt=_ts_system,
                        max_tokens=600,
                        thinking_budget=0,
                    )
                    return {
                        "response": _ts_result["text"],
                        "sources": _ts_sources,
                        "retrieval_method": "name_search:tariff_schedule_vector_fallback",
                    }

        chunks = retrieve_context(latest_question, top_k=top_k, category=effective_category)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=build_context_block(chunks))
        sources = [
            {
                "notification_no": c.get("notification_no"),
                "title": c.get("title"),
                "category": c.get("category"),
                "pdf_url": c.get("pdf_url"),
                "score": c.get("score"),
            }
            for c in chunks
        ]
        retrieval_method = "vector_search"

    llm_messages = [
        {"role": ("user" if msg["role"] == "user" else "assistant"), "content": msg["content"]}
        for msg in trim_history(messages)
    ]

    result = call_gemini(messages=llm_messages, system_prompt=system_prompt, max_tokens=1000)
    print(f"🔍 Gemini usage (single-entity, {retrieval_method}): {result['usage']}")
    response_text = result["text"]
    finish_reason = result["finish_reason"]
    if finish_reason and finish_reason not in ("stop", "STOP"):
        print(f"⚠ Gemini response did not finish cleanly — finish_reason={finish_reason}")
        response_text += (
            f"\n\n*(Note: this response was cut short by the model — finish_reason: "
            f"{finish_reason}. Try asking a more specific or shorter question.)*"
        )

    return {
        "response": response_text,
        "sources": sources,
        "retrieval_method": retrieval_method,
    }


def answer_question(query: str, top_k: int = 5, category: str = None) -> dict:
    """Convenience wrapper for a single-turn question — used by the CLI below."""
    return chat_with_context([{"role": "user", "content": query}], top_k=top_k, category=category)


if __name__ == "__main__":
    print("RegulAI RAG — type a question (Ctrl+C to quit)\n")
    while True:
        try:
            q = input("❓ ").strip()
            if not q:
                continue
            result = answer_question(q)
            print(f"\n💬 {result['response']}\n")
            print("📎 Sources:")
            for s in result["sources"]:
                title = (s.get("title") or "")[:60]
                print(f"   - {s.get('notification_no')} | {title}")
            print()
        except KeyboardInterrupt:
            print("\n👋 Bye")
            break