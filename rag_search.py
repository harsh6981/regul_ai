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
import sys
# Configure stdout/stderr to UTF-8 encoding on startup to prevent Windows console / redirect charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import re
from datetime import datetime
from pymongo import MongoClient
from google import genai
from google.genai import types
from dotenv import load_dotenv

from embed_pipeline import embed_query, DB_NAME, CHUNK_COLLECTION, MONGO_URI

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL    = "gemini-2.5-flash-lite"   # switched from gemini-2.5-flash — same API key/
# project, but a DIFFERENT model has its OWN separate daily quota pool on the free tier.
# gemini-2.5-flash's daily allowance runs out fast under real chat traffic; flash-lite gets
# a materially higher daily cap for the same $0 cost, and is already what hsn_extract_
# pipeline.py uses for LLM cleanup calls. Swap back to "gemini-2.5-flash" any time you have
# quota headroom and want slightly stronger synthesis on hard multi-doc questions.
VECTOR_INDEX    = "vector_index"
NOTIFICATIONS_COLLECTION = "notifications"

# ── LLM provider switch ──────────────────────────────────────
# Temporary escape hatch for days Gemini's free-tier quota is exhausted:
# set LLM_PROVIDER=groq in .env to route every call_gemini() call through
# Groq instead, with ZERO other code changes needed anywhere in this file
# or in app.py — every call site uses call_gemini()'s default model, so
# swapping the provider here is the only edit required. Delete/comment
# out LLM_PROVIDER in .env (or set it back to "gemini") to switch back;
# the Gemini path is untouched and still the default.
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "gemini").lower()
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL   = "https://api.groq.com/openai/v1"
# openai/gpt-oss-120b is Groq's current recommended general-purpose model —
# they deprecated llama-3.3-70b-versatile (their old default) in June 2026.
# Swap to "openai/gpt-oss-20b" for an even faster/cheaper option if 120b's
# free-tier limits get tight too.
GROQ_MODEL      = "openai/gpt-oss-120b"

_mongo_client = None
_genai_client = None
_groq_client = None


def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file")
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def get_groq_client():
    """Lazy OpenAI-SDK client pointed at Groq's OpenAI-compatible endpoint —
    same pattern hsn_extract_pipeline.py already uses for Gemini's
    OpenAI-compatible endpoint, just a different base_url/key."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set — add it to your .env file")
        from openai import OpenAI  # local import: only needed when LLM_PROVIDER=groq
        _groq_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return _groq_client


def call_groq(messages: list, system_prompt: str, max_tokens: int, temperature: float = 0.3,
              model: str = None) -> dict:
    """Groq equivalent of call_gemini() — same return shape ({"text",
    "finish_reason", "usage"}) so callers never need to know which
    provider actually served the request."""
    client = get_groq_client()
    groq_messages = [{"role": "system", "content": system_prompt}]
    groq_messages += [{"role": m["role"], "content": m["content"]} for m in messages]

    response = client.chat.completions.create(
        model=model or GROQ_MODEL,
        messages=groq_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    choice = response.choices[0]
    text = choice.message.content or ""
    finish_reason = choice.finish_reason  # already a plain string, e.g. "stop", "length"
    usage = getattr(response, "usage", None)

    return {"text": text, "finish_reason": finish_reason, "usage": usage}


def call_gemini(messages: list, system_prompt: str, max_tokens: int, temperature: float = 0.3,
                 model: str = None, thinking_budget: int = 0) -> dict:
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
    if LLM_PROVIDER == "groq":
        return call_groq(messages, system_prompt, max_tokens, temperature, model=model)

    model = model or GEMINI_MODEL
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
    r'note|footnote)\b', re.IGNORECASE
)


def looks_like_vague_followup(query: str) -> bool:
    """
    True if this message is short/generic enough that it's very unlikely
    to be a fresh, self-contained question — "what is description", "give
    me more info on that", "and this one" — rather than a new topic.

    CRITICAL: returns False immediately if the query itself contains an
    explicit chapter number or HSN code — even if short, "chapter 29" is
    NOT a vague follow-up; it's a new direct lookup. Without this guard,
    "tell me about chapter 29" (after a chapter 39 query) was being treated
    as a vague follow-up, then find_recent_entity_followup() re-fetched
    chapter 39 and used its text instead, producing "I don't have info on
    chapter 29" even though chapter 29 was explicitly named.
    """
    # If the message explicitly names a chapter or HSN code, it's a new
    # direct query — not a follow-up on whatever came before.
    if extract_chapter_numbers(query) or extract_hsn_codes(query):
        return False
    # If the message has explicit intent to find a product's duty/code
    # (e.g. "import duty on wire", "HSN code for Para-aramid Fibre"),
    # it is a FRESH question even if short — never treat it as a followup.
    # Without this, "what is the import duty on wire" (≤14 words, contains
    # FOLLOWUP_HINT_RE matches for "duty"/"rate") was classified as a vague
    # followup, which skipped the Tariff-Schedule keyword/name search and
    # fell through to unscoped vector search returning unrelated results.
    if looks_like_name_search(query):
        return False
    words = query.strip().split()
    if not words:
        return False
    # If the query contains a specific keyword indicating a followup, it's a followup
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
# Name-based comparison ("compare polyethylene and pvc")
# ─────────────────────────────────────────────────────────────
# BUG THIS FIXES: extract_hsn_codes()/extract_chapter_numbers() above
# only ever fire when the user names actual digits — a comparison
# phrased with plain product names ("compare polyethylene and pvc")
# has none, so it used to fall all the way through to the bare-name
# fallback and get treated as ONE garbled search term (the literal
# string "compare polyethylene and pvc"), which only loosely matched a
# grab-bag of unrelated rows. Worse, a short vague follow-up on that
# comparison ("compare duty rate", "make it a table") was then
# silently swallowed by the SINGLE-entity vague-followup path further
# down, which only remembers the ONE most recently mentioned HSN code
# in the whole conversation and re-answers with just that — dropping
# the second product and the comparison intent entirely, and (since
# that path is templated/zero-LLM) repeating the exact same single-code
# answer verbatim no matter how the follow-up is reworded.
COMPARISON_TRIGGER_RE = re.compile(r'\bcompare\b|\bvs\.?\b|\bversus\b', re.IGNORECASE)
COMPARISON_SPLIT_RE = re.compile(r'\s*(?:\band\b|\bwith\b|\bvs\.?\b|\bversus\b|,)\s*', re.IGNORECASE)
COMPARISON_BOILERPLATE_RE = re.compile(
    r'^\s*compare\s+|'
    r'\b(?:import\s+|customs\s+)?duty\s+rate\b|'
    r'\b(?:import|customs)\s+duty\b|'
    r'\bduty\b|\brate\b|\btariff\b|\bhsn\b|\bcode\b|'
    r'\bmake\s+(?:it\s+)?(?:a\s+|in\s+)?table\b|'
    r'\bin\s+tabular\s+form\b|\bas\s+a\s+table\b|\btable\s+format\b|'
    r'\bof\s+both(?:\s+the)?\b|\bboth\s+the\b|\bboth\b',
    re.IGNORECASE
)


def extract_comparison_terms(query: str) -> list:
    """
    Pulls the 2+ product/material NAMES being compared out of a
    name-based comparison question, e.g. "compare polyethylene and
    pvc" -> ["polyethylene", "pvc"], "compare polyethylene and pvc
    duty rate make table" -> ["polyethylene", "pvc"].

    This is the name-based counterpart to the numeric multi-entity
    check (extract_hsn_codes/extract_chapter_numbers): only trusted
    when it survives to 2+ non-trivial terms after stripping
    boilerplate — a single leftover term (or none) means this wasn't
    really expressing a comparison and the caller should fall through
    to other routing instead of guessing.
    """
    if not COMPARISON_TRIGGER_RE.search(query):
        return []
    cleaned = COMPARISON_BOILERPLATE_RE.sub(' ', query)
    parts = [p.strip(" ?.!:;,") for p in COMPARISON_SPLIT_RE.split(cleaned)]
    terms = [p for p in parts if len(p) >= 3 and not p.isdigit()]
    seen, out = set(), []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out if len(out) >= 2 else []


def find_recent_comparison_terms(messages: list) -> list:
    """
    Scans backwards through prior user turns (excluding the latest) for
    the most recent message that itself named a 2+-term product
    comparison (extract_comparison_terms), for use when the current
    message is a vague follow-up ("compare duty rate", "make it a
    table") referencing that comparison rather than restating the
    names. Mirrors find_recent_multi_entity_followup() but for
    name-based (not code/chapter-based) comparisons.
    """
    for m in reversed(messages[:-1]):
        if m.get("role") != "user":
            continue
        terms = extract_comparison_terms(m.get("content", ""))
        if terms:
            return terms
    return []


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
    r'^(?:'
    r'(?:give\s+me|what\s+is|what\s+are|what\'?s|find|show\s+(?:me\s+)?|tell\s+me|search\s+(?:for\s+)?|look\s+up|details\s+(?:of|for|about)?|info\s+(?:on|about|for)?|information\s+(?:on|about|for)?)\s+'
    r'(?:the\s+|a\s+)?'
    r'(?:import\s+|customs\s+)?'
    r'(?:duty|rate|tariff|classification|hsn(?:\s+code)?|tariff(?:\s+code)?|code|gst|bcd|exemption|concession)'
    r'(?:\s+rate|\s+of\s+duty)?'
    r'\s+(?:on|of|for|about|in|to)\s+'
    r'|'
    r'(?:give\s+me\s+(?:information|details|info)\s+(?:on|about|for)\s+|'
    r'(?:information|details|info)\s+(?:on|about|for|of)\s+|'
    r'(?:tell\s+me|show\s+me|find|search\s+for|look\s+up)\s+(?:about\s+)?|'
    r'what\s+(?:is|are|s)\s+(?:the\s+)?|'
    r'import\s+duty\s+(?:on|of|for)\s+|'
    r'duty\s+rate\s+(?:on|of|for)\s+|'
    r'rate\s+of\s+duty\s+(?:on|of|for)\s+|'
    r'hsn\s+code\s+(?:for|of|on)\s+|'
    r'tariff\s+code\s+(?:for|of|on)\s+)'
    r')\s*[:\-]?\s*(.+)',
    re.IGNORECASE,
)

SUFFIX_CLEAN_RE = re.compile(
    r'\s+(?:'
    r'(?:give\s+me|what\s+is|what\s+are|what\'?s|find|show\s+(?:me\s+)?|tell\s+me|search\s+(?:for\s+)?|look\s+up|details\s+(?:of|for|about)?|info\s+(?:on|about|for)?|information\s+(?:on|about|for)?)\s+'
    r'(?:the\s+|a\s+)?'
    r'(?:import\s+|customs\s+)?'
    r'(?:duty|rate|tariff|classification|hsn(?:\s+code)?|tariff(?:\s+code)?|code|gst|bcd|exemption|concession)'
    r'(?:\s+rate|\s+of\s+duty)?'
    r'|'
    r'(?:give\s+me\s+(?:information|details|info)\s+(?:on|about|for)?|'
    r'(?:information|details|info)\s+(?:on|about|for|of)?|'
    r'(?:tell\s+me|show\s+me|find|search\s+for|look\s+up)\s+(?:about)?|'
    r'what\s+(?:is|are|s)\s+(?:the)?|'
    r'import\s+duty|'
    r'duty\s+rate|'
    r'rate\s+of\s+duty|'
    r'hsn\s+code|'
    r'tariff\s+code|'
    r'gst\s+rate|'
    r'bcd\s+rate|'
    r'gst|'
    r'bcd|'
    r'duty|'
    r'rate|'
    r'exemption'
    r')'
    r')$',
    re.IGNORECASE
)

_NAME_SEARCH_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "is", "what", "code", "hsn",
    "import", "export", "duty", "rate", "item", "items", "product", "products",
    # Tariff-description boilerplate — these appear in a huge fraction of
    # ALL hsn_codes rows ("whether or not", "not elsewhere specified",
    # "put up for retail sale", etc.), so treating them as signal words in
    # the word-fallback OR match below produces false "matches": e.g.
    # "Edible fruit or nut trees, grafted or not" was matching random
    # organic-chemistry codes purely because their descriptions also
    # contained the word "not" — the only word in common.
    "not", "other", "others", "whether", "elsewhere", "specified", "put",
    "including", "excluding", "used", "with", "without", "than", "thereof",
    "based", "containing", "namely", "having", "more", "less", "such",
    "any", "all", "etc",
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
    cleaned_term = SUFFIX_CLEAN_RE.sub('', term).strip(" ?.!:;,")
    return cleaned_term or None


def build_loose_phrase_pattern(term: str):
    """
    Builds a regex matching `term` as a phrase, but treating ANY run of
    whitespace between words as equivalent — including newlines, not
    just a single literal space.

    This matters specifically for matching against RAW PDF-extracted
    text (notifications.full_text, chunks.text) rather than the cleaned
    hsn_codes.description field. rule_based_extract.py's own docstring
    documents that PyMuPDF's get_text() puts every tariff-table field
    (code, each description word, unit, rate...) on its OWN physical
    line. hsn_extract_pipeline.py/rule_based_extract.py reconstruct a
    clean, space-joined description before it ever reaches hsn_codes,
    but the raw full_text/chunk text used by the exact-keyword fallback
    below is untouched — so a plain re.escape("neem seeds") pattern
    requires a literal space between "neem" and "seeds", which never
    occurs in that raw text (it's actually "Neem\\nseed(s)\\n..."). This
    was the real cause behind repeated "documents don't contain this"
    answers for genuinely-present multi-word product names (e.g. "neem
    seeds", "poppy straw") whose chapter hadn't been structurally
    extracted into hsn_codes yet — the keyword fallback that was
    supposed to catch exactly this case never matched anything.

    Word-boundaries (\\b) are kept at each end so this still can't match
    partway inside an unrelated longer word. Each word also tolerates an
    optional trailing "s" (singular/plural) in either direction — e.g.
    query "neem seeds" will match text containing "Neem seed" too, since
    official tariff nomenclature doesn't reliably match a shopper's
    everyday singular/plural phrasing.
    """
    words = [w for w in term.split() if w]
    if not words:
        return None

    def word_pattern(w: str) -> str:
        base = w[:-1] if w.lower().endswith('s') and len(w) > 3 else w
        return re.escape(base) + r's?'

    return re.compile(r'\b' + r'\s+'.join(word_pattern(w) for w in words) + r'\b', re.IGNORECASE)


def search_hsn_by_name(term: str, limit: int = 15):
    """
    Finds HSN rows whose description matches `term`. Returns
    (docs, quality) where quality is one of:
      "exact"     - a row's description IS the term (case-insensitive)
      "phrase"    - the term appears verbatim as a substring somewhere
      "all_words" - every significant word in the term appears in the
                    description (not necessarily contiguous)
      "partial"   - 2+ (but not all) significant words matched
      "loose"     - only 1 significant word matched -- lowest confidence,
                    caller should show a caveat
      "none"      - nothing found at all (docs will be [])

    Tiered strategy:
      1. Whole-phrase substring match (handles "ferro-nickel", "natural
         rubber latex" etc. matching as-typed). Among these, an exact
         full-description match is ranked first, then shorter
         descriptions before longer/qualified ones -- e.g. a bare
         "Poppy straw" row should outrank "Concentrates of poppy straw"
         for a query of just "poppy straw", since the longer one
         describes a different (derived/processed) product.
      2. If no substring hit, try requiring EVERY significant word
         (3+ letters, stopwords excluded) to appear in the description --
         higher precision than a plain OR before falling back further.
      3. If that finds nothing and there are 2+ words, relax to 2+
         matched words (previous behavior).
      4. If STILL nothing, relax to a single matched word rather than
         returning empty. This matters when one of the query's words is
         a colloquial/common term that never appears in the official
         tariff nomenclature -- e.g. "mango seed": the actual tariff
         description is "Mango kernel", so "seed" never matches
         anything, but "mango" alone correctly identifies the right
         row. Requiring 2+ words unconditionally (the old behavior)
         silently dropped this and fell through to an unrelated,
         confidently-wrong vector-search answer instead. Because this
         tier is lower-confidence, callers should surface a caveat
         rather than presenting it as an exact match.
    """
    col = get_hsn_collection()
    term = term.strip()
    if not term:
        return [], "none"

    def is_exact(doc):
        return (doc.get("description") or "").strip().lower() == term.lower()

    whole_pattern = re.compile(re.escape(term), re.IGNORECASE)
    phrase_docs = list(col.find({"description": whole_pattern}, {"_id": 0}).limit(limit * 3))
    if phrase_docs:
        # Exact full-description matches first, then shortest description
        # first (closer to the bare term = more likely the literal/base
        # item rather than a derivative/concentrate/preparation of it).
        phrase_docs.sort(key=lambda d: (not is_exact(d), len(d.get("description") or "")))
        quality = "exact" if any(is_exact(d) for d in phrase_docs) else "phrase"
        return phrase_docs[:limit], quality

    words = [w for w in re.findall(r"[A-Za-z\-]{3,}", term) if w.lower() not in _NAME_SEARCH_STOPWORDS]
    if not words:
        return [], "none"

    # Use word-boundary anchors (\b) so "live" matches the word "live" but
    # NOT "liver", "livestock", etc. -- the old plain-substring pattern was
    # returning completely wrong HSN codes for searches like "live animals"
    # (matching "Liquid extracts of liver" via the "live" -> "liver" substring).
    word_patterns = [re.compile(r'\b' + re.escape(w) + r'\b', re.IGNORECASE) for w in words]
    candidates = list(col.find({"$or": [{"description": p} for p in word_patterns]}, {"_id": 0}).limit(limit * 4))

    def relevance(doc):
        desc = doc.get("description", "")
        return sum(1 for p in word_patterns if p.search(desc))

    all_words_docs = [d for d in candidates if relevance(d) == len(word_patterns)]
    if all_words_docs:
        all_words_docs.sort(key=relevance, reverse=True)
        return all_words_docs[:limit], "all_words"

    if len(word_patterns) > 1:
        strong = [d for d in candidates if relevance(d) >= 2]
        if strong:
            strong.sort(key=relevance, reverse=True)
            return strong[:limit], "partial"

    loose = [d for d in candidates if relevance(d) >= 1]
    if loose:
        loose.sort(key=relevance, reverse=True)
        return loose[:limit], "loose"

    return [], "none"


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


_NAME_SEARCH_CAVEATS = {
    "all_words": None,
    "phrase": None,
    "exact": None,
    "partial": (
        "**Note:** this isn't an exact match — the official tariff description "
        "doesn't contain every word of your search term, so double-check this is "
        "the item you meant before relying on the rate below."
    ),
    "loose": (
        "**Note:** no close match was found for your exact wording, so this is the "
        "*closest* tariff entry based on a partial word match — it may not be the "
        "precise item you meant. If this looks wrong, try the chapter number "
        "directly, or rephrase using the official tariff terminology."
    ),
}


def format_hsn_name_search_answer(term: str, docs: list, quality: str = "phrase") -> str:
    """Deterministic, zero-LLM answer listing every HSN row whose
    description matched the searched-for name. Shows full duty/rate
    data and uses a table when multiple results are found.

    `quality` (from search_hsn_by_name's tiered matcher) controls
    whether a caveat is prepended — "partial"/"loose" matches are shown
    with an explicit disclaimer rather than presented with the same
    confidence as an exact match, since they're more likely to be the
    wrong product (see search_hsn_by_name's docstring for the "mango
    seed" / "poppy straw" style failures this guards against)."""
    if not docs:
        return (
            f"No HSN code matching \"{term}\" was found in the chapters processed so far. "
            f"This could mean the term doesn't appear verbatim in any tariff description, or "
            f"that the relevant chapter hasn't been run through the extraction pipeline yet — "
            f"try a more specific or differently-worded term, or look it up by chapter/code "
            f"directly if you know either one."
        )

    caveat = _NAME_SEARCH_CAVEATS.get(quality)
    caveat_prefix = (caveat + "\n\n") if caveat else ""

    if len(docs) == 1:
        # Single result — rich card-style format with all fields
        d = docs[0]
        code = d.get("hsn_code", "N/A")
        desc = d.get("description") or "(no description on file)"
        unit = d.get("unit")
        std = d.get("standard_rate") or "not specified"
        pref = d.get("preferential_rate")
        footnote = d.get("footnote_text")
        chapter = d.get("chapter")
        lines = [f"**HSN Code {code}** — {desc}"]
        if unit:
            lines.append(f"**Unit:** {unit}")
        lines.append(f"**Standard Rate of Duty:** {std}")
        if pref:
            lines.append(f"**Preferential Areas Rate:** {pref}")
        if footnote:
            lines.append(f"**Note:** {footnote}")
        if chapter:
            lines.append(f"*(Tariff Schedule, Chapter {chapter})*")
        return caveat_prefix + "\n".join(lines)

    # Multiple results — full markdown table with all columns
    header  = "| HSN Code | Description | Unit | Standard Rate | Preferential Rate | Chapter |"
    divider = "|---|---|---|---|---|---|"
    rows = [
        f"Found **{len(docs)}** HSN code{'s' if len(docs) != 1 else ''} matching **\"{term}\"**:\n",
        header,
        divider,
    ]
    footnotes = []
    for d in docs:
        code  = d.get("hsn_code", "N/A")
        desc  = (d.get("description") or "(no description)").replace("|", "/")
        unit  = d.get("unit") or "—"
        std   = d.get("standard_rate") or "—"
        pref  = d.get("preferential_rate") or "—"
        chap  = d.get("chapter") or "—"
        rows.append(f"| {code} | {desc} | {unit} | {std} | {pref} | {chap} |")
        if d.get("footnote_text"):
            footnotes.append(f"- **{code}**: {d['footnote_text']}")

    out = "\n".join(rows)
    if footnotes:
        out += "\n\n**Notes:**\n" + "\n".join(footnotes)
    out += "\n\n*Ask for any of these codes directly for more details, or ask me to compare a few.*"
    return caveat_prefix + out


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


# ─────────────────────────────────────────────────────────────
# Duty-rate TREND lookup — "how has the duty rate on X changed over time"
# ─────────────────────────────────────────────────────────────
# There is no versioned/historical rate table anywhere in this pipeline —
# hsn_codes stores only the CURRENT rate as extracted from today's tariff
# PDFs. What DOES already carry real dates is the 'notifications'
# collection (individual amendment/exemption notifications, scraped with
# their actual issue dates by project.py). So a trend answer is built by:
#   1. Resolving the target HSN code (from an explicit code, a product
#      name via search_hsn_by_name, or a follow-up reference).
#   2. Pulling the current rate + any "w.e.f. <date>" footnote from
#      hsn_codes (already captures some effective dates, e.g. see the
#      "Concentrates of poppy straw" example: "w.e.f. 1.5.2022").
#   3. Searching 'notifications' for any doc whose full_text mentions
#      that code (in its various written forms), sorted chronologically.
#   4. Handing all of that to Gemini with an explicit instruction to
#      report ONLY dates/rates that are stated verbatim in the retrieved
#      text — never to interpolate or guess a trend that isn't actually
#      documented. If nothing turns up, the answer says so plainly
#      instead of fabricating a history.
TREND_INTENT_RE = re.compile(
    r'\b(trend|trends|history|historical|over\s+the\s+years|over\s+time|'
    r'how\s+has.{0,40}chang|rate\s+chang|changed\s+over|past\s+rates?|'
    r'previous\s+rates?|rate\s+history|used\s+to\s+be)\b',
    re.IGNORECASE,
)


def looks_like_trend_query(query: str) -> bool:
    """True if the question is asking how a duty rate has changed over
    time, rather than just what it is right now."""
    return bool(TREND_INTENT_RE.search(query))


def extract_trend_target_term(query: str) -> str:
    """
    Pulls the product/HSN name out of a trend-style question, e.g.
    "show me the duty rate trend for poppy straw" -> "poppy straw".
    Falls back to the whole (cleaned) query if no "for/of/on X" tail is
    found, so a bare "trend for HSN 2939 11 00" still leaves something
    for the caller to try extract_hsn_codes() on.
    """
    m = re.search(r'\b(?:trend|trends|history|historical)\b.*?\b(?:for|of|on)\s+(.+)', query, re.IGNORECASE)
    term = m.group(1) if m else query
    term = SUFFIX_CLEAN_RE.sub('', term).strip(" ?.!:;,")
    term = re.sub(r'^(?:the|how|has|did|does|import|customs|duty|rate)\s+', '', term, flags=re.IGNORECASE).strip()
    return term


def find_rate_change_notifications(hsn_code: str, limit: int = 8):
    """
    Searches the 'notifications' collection (real dated amendment/
    exemption notifications, NOT the static Tariff Schedule) for any
    document whose full_text mentions this HSN code, in the handful of
    ways it's commonly written (unspaced, or split into 4/2/2-digit
    groups). Returns docs sorted oldest-first where a date could be
    parsed; unparseable dates are pushed to the end rather than
    dropped, since that's still useful context for the LLM even if it
    can't be placed on a timeline.
    """
    col = get_notifications_collection()
    code_variants = {hsn_code}
    if len(hsn_code) >= 6:
        code_variants.add(f"{hsn_code[:4]} {hsn_code[4:6]}")
    if len(hsn_code) == 8:
        code_variants.add(f"{hsn_code[:4]} {hsn_code[4:6]} {hsn_code[6:]}")

    patterns = [re.compile(r'\b' + re.escape(v) + r'\b') for v in code_variants]
    query = {
        "category": {"$ne": "Tariff Schedule"},
        "$or": [{"full_text": p} for p in patterns],
    }
    try:
        docs = list(col.find(query, {
            "_id": 0, "notification_id": 1, "notification_no": 1, "title": 1,
            "date": 1, "category": 1, "full_text": 1, "pdf_url": 1,
        }).limit(limit))
    except Exception as e:
        print(f"Trend notification search failed: {e}")
        return []

    def parse_dt(d):
        for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(d.get("date", ""), fmt)
            except Exception:
                continue
        return None

    docs.sort(key=lambda d: parse_dt(d) or datetime.max)
    return docs


TREND_SYSTEM_PROMPT_TEMPLATE = """You are RegulAI, an expert assistant on Indian customs duty rates.

The user wants to know how the duty rate for a specific HSN code/product
has changed OVER TIME, not just what it is today. Below is (a) the
current structured tariff record for the code, and (b) any notifications
found in the database that mention this code, listed in roughly
chronological order (oldest first, where a date was available).

--- DATA ---
{context}
--- END DATA ---

Build a short chronological trend summary using ONLY dates and rates that
are explicitly stated in the data above:
- For each notification that clearly states a rate change, add one bullet:
  "- <date>: rate changed to <rate> (<Notification No. or ID>)".
- Do NOT invent, estimate, or interpolate any rate or date that isn't
  explicitly present in the text above.
- If the notifications above don't clearly document historical rate
  changes for this code (e.g. they only mention the code in passing, or
  none were found at all), say so plainly, and fall back to showing just
  the CURRENT rate — including its effective date if the footnote states
  one (e.g. "w.e.f. 1.5.2022").
- Never present a guess or estimate as if it were a documented fact.
End with one sentence noting that this reflects only notifications
indexed in this database, not the complete legislative history.
"""


def build_trend_answer(target_doc: dict, messages: list) -> dict:
    """
    Builds a duty-rate trend answer for a single resolved HSN document.
    Grounds the answer in whatever dated notifications actually exist
    for this code, rather than fabricating a timeline.
    """
    hsn_code = target_doc.get("hsn_code")
    desc = target_doc.get("description") or "(no description on file)"
    current_rate = target_doc.get("standard_rate") or "not specified"
    footnote = target_doc.get("footnote_text")

    history_docs = find_rate_change_notifications(hsn_code) if hsn_code else []

    context_parts = [
        "CURRENT TARIFF RECORD (from the structured tariff schedule):\n"
        f"HSN Code: {hsn_code}\nDescription: {desc}\n"
        f"Current Standard Rate of Duty: {current_rate}"
        + (f"\nFootnote: {footnote}" if footnote else "")
    ]
    for d in history_docs:
        context_parts.append(
            f"=== Notification: {d.get('notification_no') or d.get('notification_id')} | "
            f"{(d.get('title') or '')[:80]} | Date: {d.get('date', 'N/A')} | "
            f"Category: {d.get('category', 'N/A')} ===\n{(d.get('full_text') or '')[:3000]}"
        )
    context_block = "\n\n".join(context_parts)
    system_prompt = TREND_SYSTEM_PROMPT_TEMPLATE.format(context=context_block)

    llm_messages = [
        {"role": ("user" if m["role"] == "user" else "assistant"), "content": m["content"]}
        for m in trim_history(messages)
    ]
    result = call_gemini(messages=llm_messages, system_prompt=system_prompt, max_tokens=700, thinking_budget=0)

    sources = [{
        "notification_no": d.get("notification_no"),
        "title": d.get("title"),
        "category": d.get("category"),
        "pdf_url": d.get("pdf_url"),
        "score": None,
    } for d in history_docs]

    return {
        "response": result["text"],
        "sources": sources,
        "retrieval_method": f"trend_lookup:hsn_code ({len(history_docs)} historical notifications found)",
    }


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
        compact = code.replace(" ", "")
        
        # Dynamically build standard Tariff spacing (e.g. 8 digit: 'XXXX XX XX', 6 digit: 'XXXX XX')
        if len(compact) == 8:
            spaced = f"{compact[:4]} {compact[4:6]} {compact[6:]}"
        elif len(compact) == 6:
            spaced = f"{compact[:4]} {compact[4:]}"
        else:
            spaced = compact

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
    # Normalize common spaced preposition typos (e.g. "o n" -> "on", "o f" -> "of", "f o r" -> "for")
    latest_question = re.sub(r'\bo\s+n\b', 'on', latest_question, flags=re.IGNORECASE)
    latest_question = re.sub(r'\bo\s+f\b', 'of', latest_question, flags=re.IGNORECASE)
    latest_question = re.sub(r'\bf\s+o\s+r\b', 'for', latest_question, flags=re.IGNORECASE)

    # ── Duty-rate TREND lookup ("how has the duty on X changed over time") ──
    # Checked before name search / direct lookup: a trend question like
    # "duty rate trend for HSN 2939 11 00" or "history of duty on poppy
    # straw" would otherwise get treated as an ordinary current-rate
    # lookup by the branches below, silently dropping the "over time"
    # part of the question.
    if looks_like_trend_query(latest_question):
        _trend_target_doc = None

        _trend_codes = extract_hsn_codes(latest_question)
        if _trend_codes:
            _resolved = resolve_hsn_reference(_trend_codes[0])
            if _resolved:
                _trend_target_doc = _resolved[0]

        if not _trend_target_doc:
            _trend_term = extract_trend_target_term(latest_question)
            if _trend_term:
                _trend_matches, _trend_quality = search_hsn_by_name(_trend_term)
                if _trend_matches:
                    _trend_target_doc = _trend_matches[0]

        if not _trend_target_doc:
            # e.g. "show me the trend for that" right after an HSN lookup
            _kind, _value = find_recent_entity_followup(messages)
            if _kind == "hsn_code":
                _resolved = resolve_hsn_reference(_value)
                if _resolved:
                    _trend_target_doc = _resolved[0]

        if _trend_target_doc:
            return build_trend_answer(_trend_target_doc, messages)

        return {
            "response": (
                "I couldn't tell which product or HSN code you'd like the duty-rate "
                "trend for — could you name the HSN code (e.g. \"2939 11 00\") or the "
                "product itself (e.g. \"poppy straw\")?"
            ),
            "sources": [],
            "retrieval_method": "trend_lookup:unresolved",
        }

    # ── Name-based HSN search ("HSN code for ferro-nickel", "import duty on X") ──
    # Checked first and separately from everything below: this answers a
    # fundamentally different question (name -> code) than the rest of
    # this function (code/chapter/notification-number -> details), and a
    # query like "hsn code for natural rubber" would otherwise fall
    # through every other branch and hit generic vector search, which
    # has no path back to a specific hsn_codes row. Templated/zero-LLM,
    # same rationale as the rest of the structured HSN answers.
    #
    # KEY: we also stash the extracted term + low-confidence matches into
    # _explicit_intent_term / _explicit_intent_matches so the bare-name
    # Tariff-Schedule fallback in the else-block below can use them
    # WITHOUT repeating the search. Previously, when looks_like_name_search()
    # was True but quality was "none" (no match at all), the else block's
    # _name_search_attempted flag was False (because the query had no code/
    # chapter/notification) so the chapter-text keyword search was SKIPPED
    # entirely and the query fell through to unscoped vector search over
    # non-Tariff-Schedule docs — which surfaced "Project imports",
    # "passenger's baggage", etc. for product names like "Para-aramid Fibre".
    _explicit_intent_term = None
    _explicit_intent_matches = []
    _explicit_intent_quality = "none"
    if looks_like_name_search(latest_question):
        term = extract_name_search_term(latest_question)
        if term:
            matches, match_quality = search_hsn_by_name(term)
            # Only trust this enough to answer immediately for the
            # higher-confidence tiers. "loose" (1 shared word) and
            # "partial" (2+ shared words but not all of them) are both
            # too easily wrong — e.g. "sugar cane" matching "Sugars,
            # chemically pure..." via the shared word "sugar" (loose),
            # or "durries of man-made fibres" matching unrelated
            # Chapter 55 synthetic-fibre carpet rows that only share
            # "man-made"/"fibres" (partial). Don't return here for those
            # — fall through so the hybrid RAG fallback (Tariff-Schedule
            # exact-phrase + vector search) gets a chance to find the
            # real row first; this branch's match becomes the final
            # last-resort fallback within the bare-name section further
            # down (which re-derives the same clean term and re-runs
            # this same search).
            # Always stash the term + result for the else-block fallback.
            _explicit_intent_term = term
            _explicit_intent_matches = matches
            _explicit_intent_quality = match_quality
            if matches and match_quality not in ("loose", "partial"):
                return {
                    "response": format_hsn_name_search_answer(term, matches, match_quality),
                    "sources": [{
                        "notification_no": None,
                        "title": f"HSN {d.get('hsn_code')} — {d.get('description', '')[:80]}",
                        "category": "HSN Code",
                        "pdf_url": None,
                        "score": None,
                    } for d in matches],
                    "retrieval_method": f"name_search:hsn_code (explicit intent, match={match_quality}, no LLM call)",
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

    # ── Multi-entity NAME comparison ("compare polyethylene and pvc") ──
    # Numeric code/chapter comparisons (above) take priority when both
    # are somehow present. This covers comparisons expressed as plain
    # product names instead — see extract_comparison_terms()/
    # find_recent_comparison_terms() docstrings for the bug this fixes.
    # Tries the current message first; if that's not itself a
    # comparison but reads as a vague follow-up ("compare duty rate",
    # "make it a table"), reuses the terms from the most recent prior
    # comparison instead of letting the single-entity follow-up path
    # further down silently grab just one old HSN code.
    #
    # IMPORTANT: only set multi_docs if ALL comparison terms resolve
    # confidently. If some resolve and others don't, DON'T force a
    # partial answer — instead fall through to independent name-search
    # handling below, which will handle the successful terms as
    # separate lookups. This prevents awkward half-comparisons like
    # "I found PVC, but not wheat" instead of searching both independently.
    if multi_docs is None:
        comparison_terms = extract_comparison_terms(latest_question)
        if not comparison_terms and looks_like_vague_followup(latest_question):
            comparison_terms = find_recent_comparison_terms(messages)

        if comparison_terms:
            name_docs, unmatched_terms = [], []
            for term in comparison_terms:
                term_matches, term_quality = search_hsn_by_name(term)
                if term_matches and term_quality not in ("loose", "partial"):
                    name_docs.append(term_matches[0])
                else:
                    unmatched_terms.append(term)

            if len(name_docs) >= 2 and not unmatched_terms:
                # All comparison terms resolved confidently — use the
                # multi-entity templated table path.
                multi_docs, multi_kind = name_docs, "hsn_code"
            # If some or all terms failed to resolve confidently, DON'T
            # force a partial comparison. Set multi_docs = None and let
            # the code below (bare-name fallback, then vector search)
            # handle the terms independently instead. This produces better
            # answers than "I found X but not Y" for a comparison request.

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
            "retrieval_method": f"direct_lookup:{direct_lookup_kind} (templated, no LLM call)",
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
        # Check if this is ACTUALLY a followup: looks_like_vague_followup()
        # returns True for any short query (≤6 words), which includes bare
        # product names like "ferro-nickel" or "bajra". But if there are no
        # prior user messages, it CAN'T be a followup — it's a fresh query.
        # Without this, first-message bare-name queries skipped name search
        # entirely and fell through to vector search (returning unrelated
        # results like "Project imports" for "ferro-nickel").
        _has_prior_user_msgs = any(m.get("role") == "user" for m in messages[:-1])
        _is_followup = _has_prior_user_msgs and looks_like_vague_followup(latest_question)
        # _name_search_attempted is True when the bare-name fallback SHOULD
        # run — either a fresh query with no codes/chapters (original case),
        # OR an explicit-intent query (looks_like_name_search was True) that
        # fell through because no confident match was found above. Without
        # the second clause, queries like "import duty on Para-aramid Fibre"
        # that didn't match any hsn_codes row would skip the chapter-text
        # keyword search entirely and go straight to unscoped vector search.
        _name_search_attempted = (
            not _has_code_or_chapter and not _is_followup
        ) or (
            _explicit_intent_term is not None  # explicit-intent query that fell through
        )
        if _name_search_attempted:
            # Strip common request verbiage so we search for just the product
            # name. We reuse a prefix-only version of the intent pattern
            # to make cleaning consistent with matching.
            _prefix_clean_pat = re.compile(
                r'^(?:'
                r'(?:give\s+me|what\s+is|what\s+are|what\'?s|find|show\s+(?:me\s+)?|tell\s+me|search\s+(?:for\s+)?|look\s+up|details\s+(?:of|for|about)?|info\s+(?:on|about|for)?|information\s+(?:on|about|for)?)\s+'
                r'(?:the\s+|a\s+)?'
                r'(?:import\s+|customs\s+)?'
                r'(?:duty|rate|tariff|classification|hsn(?:\s+code)?|tariff(?:\s+code)?|code|gst|bcd|exemption|concession)'
                r'(?:\s+rate|\s+of\s+duty)?'
                r'\s+(?:on|of|for|about|in|to)\s+'
                r'|'
                r'(?:give\s+me\s+(?:information|details|info)\s+(?:on|about|for)\s+|'
                r'(?:information|details|info)\s+(?:on|about|for|of)\s+|'
                r'(?:tell\s+me|show\s+me|find|search\s+for|look\s+up)\s+(?:about\s+)?|'
                r'what\s+(?:is|are|s)\s+(?:the\s+)?|'
                r'import\s+duty\s+(?:on|of|for)\s+|'
                r'duty\s+rate\s+(?:on|of|for)\s+|'
                r'rate\s+of\s+duty\s+(?:on|of|for)\s+|'
                r'hsn\s+code\s+(?:for|of|on)\s+|'
                r'tariff\s+code\s+(?:for|of|on)\s+)'
                r')',
                re.IGNORECASE
            )
            _clean_term = _prefix_clean_pat.sub('', latest_question.strip())
            _clean_term = SUFFIX_CLEAN_RE.sub('', _clean_term).strip(" ?.!:;,")

            # If the explicit-intent branch above already ran search_hsn_by_name()
            # for this query (and stashed the term), reuse that result rather
            # than running the exact same search a second time.
            if _explicit_intent_term and _explicit_intent_term == _clean_term:
                _bare_matches, _bare_quality = _explicit_intent_matches, _explicit_intent_quality
            elif _explicit_intent_term:
                # Explicit intent found a slightly different clean term (rare) —
                # prefer the explicit-intent term since it came from the purpose-
                # built intent regex rather than the generic prefix stripper.
                _clean_term = _explicit_intent_term
                _bare_matches, _bare_quality = _explicit_intent_matches, _explicit_intent_quality
            else:
                _bare_matches, _bare_quality = search_hsn_by_name(_clean_term) if _clean_term else ([], "none")

            # High-confidence structured match -> answer immediately.
            # "loose" (1 shared word) and "partial" (2+ shared words but
            # not all) are deliberately NOT trusted here — see below, and
            # see the matching comment on the explicit-intent branch
            # above for the concrete failures ("sugar cane", "durries of
            # man-made fibres", "vaccines for veterinary medicine") this
            # guards against.
            if _bare_matches and _bare_quality not in ("loose", "partial"):
                return {
                    "response": format_hsn_name_search_answer(_clean_term, _bare_matches, _bare_quality),
                    "sources": [{
                        "notification_no": None,
                        "title": f"HSN {d.get('hsn_code')} — {d.get('description', '')[:80]}",
                        "category": "HSN Code",
                        "pdf_url": None,
                        "score": None,
                    } for d in _bare_matches],
                    "retrieval_method": f"name_search:hsn_code (bare-name fallback, match={_bare_quality}, no LLM call)",
                }

            # Nothing confident in the structured hsn_codes collection.
            # A "loose" hsn_codes hit only shares ONE word with the query
            # and can be dangerously generic — e.g. "sugar cane" matching
            # "Sugars, chemically pure..." purely via the shared word
            # "sugar" — so it's held back rather than returned yet.
            #
            # Before trusting (or discarding) it, check whether the
            # exact phrase genuinely appears, verbatim, in the raw text
            # of a Tariff Schedule chapter (using the newline/plural-
            # tolerant matcher — see build_loose_phrase_pattern's
            # docstring). A real literal phrase hit in the chapter's own
            # text is far less likely to be a false positive than a
            # single shared word in an unrelated hsn_codes row, so it
            # takes priority over the "loose" structured match.
            _exact_chapter_docs = []
            _loose_pattern = build_loose_phrase_pattern(_clean_term) if _clean_term and len(_clean_term) >= 3 else None
            if _loose_pattern:
                try:
                    _col_notif = get_notifications_collection()
                    _exact_chapter_docs = list(_col_notif.find({
                        "category": "Tariff Schedule",
                        "full_text": _loose_pattern
                    }, {"notification_id": 1}))
                except Exception as _e:
                    print(f"Keyword search failed: {_e}")

            _ts_chunks = []
            _ts_source_kind = "none"
            if _exact_chapter_docs:
                # Genuine literal phrase match in a chapter's raw text —
                # pull the matching chunk(s) directly.
                try:
                    _chunks_col = get_chunks_collection()
                    _ts_chunks = list(_chunks_col.find({
                        "notification_id": {"$in": [d["notification_id"] for d in _exact_chapter_docs]},
                        "text": _loose_pattern
                    }, {
                        "_id": 0, "text": 1, "notification_id": 1, "title": 1,
                        "category": 1, "notification_no": 1, "date": 1, "pdf_url": 1
                    }).limit(3))
                    if _ts_chunks:
                        _ts_source_kind = "exact_chunk"
                except Exception as _e:
                    print(f"Keyword chunks retrieval failed: {_e}")

                if not _ts_chunks:
                    # We CONFIRMED the phrase is in this chapter's raw
                    # full_text, but no embedded chunk contains it —
                    # either this chapter was scraped (mongo_pipeline.py)
                    # but never (re-)embedded (embed_pipeline.py), or a
                    # chunk boundary happened to split the phrase across
                    # two chunks. Don't silently discard a CONFIRMED
                    # match and fall back to unrelated vector results —
                    # build a synthetic "chunk" directly from the
                    # chapter's own full_text: a window of raw text
                    # centered on the match (shaped like a real chunk
                    # dict so it flows into build_context_block()
                    # unchanged). This is exactly the scenario a heading-
                    # only phrase like "Parts and accessories for X"
                    # produces: the row's own stored description is just
                    # the terse child text (e.g. "Of X") per
                    # hsn_extract_pipeline.py's extraction prompt, which
                    # explicitly does NOT fold parent heading context
                    # into child descriptions — but the RAW page text
                    # still has the heading immediately followed by the
                    # row, so a windowed excerpt captures both.
                    try:
                        _full_docs = list(get_notifications_collection().find(
                            {"notification_id": {"$in": [d["notification_id"] for d in _exact_chapter_docs]}},
                            {"_id": 0, "notification_id": 1, "full_text": 1, "title": 1,
                             "category": 1, "notification_no": 1, "date": 1, "pdf_url": 1}
                        ))
                        for _d in _full_docs:
                            _ft = _d.get("full_text") or ""
                            _m = _loose_pattern.search(_ft)
                            if not _m:
                                continue
                            _window_start = max(0, _m.start() - 800)
                            _window_end = min(len(_ft), _m.end() + 800)
                            _ts_chunks.append({
                                "text": _ft[_window_start:_window_end],
                                "notification_id": _d.get("notification_id"),
                                "title": _d.get("title"),
                                "category": _d.get("category"),
                                "notification_no": _d.get("notification_no"),
                                "date": _d.get("date"),
                                "pdf_url": _d.get("pdf_url"),
                                "score": None,
                            })
                        if _ts_chunks:
                            _ts_source_kind = "synthetic_chunk"
                            print(f"Built synthetic chunk(s) from full_text for: "
                                  f"{[d.get('notification_id') for d in _exact_chapter_docs]}")
                    except Exception as _e:
                        print(f"Synthetic chunk build failed: {_e}")

            if not _ts_chunks and not _bare_matches and _clean_term and len(_clean_term) >= 3:
                # No exact phrase hit either — last resort is a broad
                # semantic search scoped to Tariff Schedule (still far
                # better than an unscoped vector search across every
                # notification, which is what surfaced "Project imports"
                # for unrelated product names in earlier testing).
                _ts_chunks = retrieve_context(_clean_term, top_k=3, category="Tariff Schedule")
                if _ts_chunks:
                    _ts_source_kind = "semantic_only"

            if _ts_chunks:
                    _ts_context = build_context_block(_ts_chunks)
                    _ts_system = SYSTEM_PROMPT_TEMPLATE.format(context=_ts_context)
                    _ts_system += (
                        f"\n\nThe user is looking for the HSN/tariff code and duty rate "
                        f"for \"{_clean_term}\". Search the tariff table text above for "
                        f"the entry whose description matches or is the general/parent "
                        f"heading for \"{_clean_term}\" (tariff schedules classify by broad "
                        f"category, not by every specific variant/colour/grade name — e.g. "
                        f"a specific dye shade falls under its general heading like "
                        f"'Disperse dyes and preparations based thereon', not its own line). "
                        f"ALWAYS answer using this exact markdown table format, even for a "
                        f"single row:\n\n"
                        f"| HSN Code | Description | Unit | Standard Rate | Preferential Rate | Chapter |\n"
                        f"|---|---|---|---|---|---|\n"
                        f"| ... | ... | ... | ... | ... | ... |\n\n"
                        f"If you match the general/parent heading rather than the exact term, "
                        f"add one line below the table: \"Note: matched under the general "
                        f"heading above — '{_clean_term}' is not broken out as its own tariff "
                        f"line.\" Only if the text above truly has nothing relevant should you "
                        f"skip the table and say so plainly, suggesting which chapter to check."
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
                        "retrieval_method": (
                            f"name_search:tariff_schedule_vector_fallback "
                            f"(source={_ts_source_kind}, hsn_codes_match={_bare_quality})"
                        ),
                    }

            # Absolute last resort: nothing confident anywhere, but we DO
            # have a low-confidence single-word hsn_codes match stashed
            # from above — better to show it with a clear caveat than to
            # fall through to unscoped vector search, which has
            # historically returned completely unrelated notification
            # docs for product-name queries ("bajra" -> "Project imports").
            if _bare_matches:
                return {
                    "response": format_hsn_name_search_answer(_clean_term, _bare_matches, _bare_quality),
                    "sources": [{
                        "notification_no": None,
                        "title": f"HSN {d.get('hsn_code')} — {d.get('description', '')[:80]}",
                        "category": "HSN Code",
                        "pdf_url": None,
                        "score": None,
                    } for d in _bare_matches],
                    "retrieval_method": f"name_search:hsn_code (bare-name fallback, match={_bare_quality}, no LLM call)",
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
        retrieval_method = "vector_search (after exhausted name search)" if _name_search_attempted else "vector_search"

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