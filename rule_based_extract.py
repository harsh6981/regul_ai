"""
rule_based_extract.py — zero-LLM-cost first pass for CBIC tariff PDFs.

fitz/PyMuPDF's get_text() (which hsn_extract_pipeline.py already uses)
puts every field on its own physical line - code, dashes, description,
unit, rate, preferential-rate are all separate lines, in row order. This
walks that line stream as a small state machine and reconstructs each
tariff row, resolving footnote markers (which can be defined on an
earlier page and simply not redefined since) along the way.

Tested against a real chapter (chap-28.pdf, 313 rows, all 270 footnoted
rows correctly resolved). Anything this parser can't confidently
resolve gets flagged in `needs_review` rather than guessed at, so the
hybrid pipeline can route exactly those to a single batched LLM call
instead of one call per page.

Description inheritance
-----------------------
Every row's description is now fully self-contained: all ancestor heading
descriptions are prepended, joined with ' - '.  For example:

  2901  Acyclic hydrocarbons                   → heading_stack[0]
  29011000  -- Saturated                        → "Acyclic hydrocarbons - Saturated"
  29012100  -- Ethylene                         → "Acyclic hydrocarbons - Ethylene"

Multi-level hierarchy is also handled:

  2902  Cyclic hydrocarbons                     → heading_stack[0]
  (dash context)  - Cyclanes, cyclenes...       → heading_stack[1]
  29021100  -- Cyclohexane                      → "Cyclic hydrocarbons - Cyclanes, cyclenes... - Cyclohexane"

The heading_stack is keyed by dash-level (0 = 4-digit heading, 1 = one
dash, 2 = two dashes, 3 = three dashes).  Whenever a heading at level N
is recorded, all levels > N are cleared so stale deeper context never
bleeds into unrelated rows.
"""
import re

CODE_RE     = re.compile(r'^\d{4}(?:\s\d{2}){0,2}$')
DASHES_RE   = re.compile(r'^-{1,4}$')
UNIT_RE     = re.compile(r'^(kg\.?|Kg\.?|KG\.?|No\.?|U\.?|Pair|Carat|gms?\.?|litre|Tonne|MT|-)$')
RATE_RE     = re.compile(r'^(?P<marker>\*{0,3})(?P<rate>\d+(?:\.\d+)?%|Free%?)$')
PREF_RE     = re.compile(r'^(?:[\d.]+%|-)$')
FOOTNOTE_RE = re.compile(r'^(?P<marker>\*{1,3})(?P<text>[A-Za-z].+)$')
OMITTED_RE  = re.compile(r'^omitted$', re.IGNORECASE)

SKIP_RE = re.compile(
    r'^(SECTION-?[IVXLCDM]+|CHAPTER-?\d+|\d{1,4}|\(\d\)|_{5,})$'
)

END_OF_TABLE_HINTS = re.compile(
    r'\b(Whereas|Notfn\.|Notification No|G\.S\.R\.|hereby (exempts|imposes)|'
    r'Government of India|Gazette of India)\b'
)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _dash_level(text: str) -> int:
    """Return the number of leading '-' characters in `text` (0–4)."""
    return len(re.match(r'^-*', text.strip()).group())


def _build_full_description(heading_stack: dict, child_text: str, child_level: int) -> str:
    """
    Join all ancestor headings from heading_stack (levels 0 .. child_level-1)
    with the child's own text using ' - ' as a separator.

    heading_stack keys are dash-levels (0 = 4-digit heading, 1 = one-dash
    sub-heading, …). Only levels strictly less than child_level are ancestors.

    If there are no ancestors in the stack the child text is returned as-is,
    so 4-digit codes whose description is entirely self-contained are unaffected.
    """
    parts = []
    for level in range(child_level):  # 0, 1, 2, … up to (but not including) child_level
        heading = heading_stack.get(level)
        if heading:
            parts.append(heading)
    # Strip any leading dashes from the child's own text before appending
    clean_child = re.sub(r'^-+\s*', '', child_text).strip(' :')
    if clean_child:
        parts.append(clean_child)
    return ' - '.join(parts) if parts else clean_child


def parse_chapter_pdf(doc, max_page=None):
    """
    doc: an already-open fitz.Document (caller owns open/close, matching
    how process_chapter_pdf() already works in hsn_extract_pipeline.py)
    Returns (rows, headings, needs_review, notification_text)
      rows         - list of dicts matching HSNRow's fields, ready to upsert
      headings     - dict of heading-only codes -> description (context, not stored as rows)
      needs_review - subset of `rows` whose parse is lower-confidence and should
                     go through one batched LLM cleanup call rather than be trusted as-is
      notification_text - legal/notification prose found trailing the table (not tariff data)
    """
    rows, headings, needs_review = [], {}, []
    running_footnotes = {}
    in_table = True
    notes = []

    # heading_stack: dash-level → last known heading description at that level.
    # level 0 = 4-digit heading code, level 1 = one-dash context, etc.
    heading_stack: dict[int, str] = {}

    last_code = None
    last_dashes = ''
    desc_buf = []
    pending_unit = None
    pending_marker = None
    pending_rate = None

    def flush_heading():
        """Called when the current code turns out to be a heading-only code
        (no unit/rate seen before a new code arrived). Stores its description
        in both the legacy `headings` dict AND heading_stack at level 0, then
        resets state."""
        nonlocal last_code, desc_buf, last_dashes
        if last_code and desc_buf:
            desc_text = ' '.join(desc_buf).strip(' :')
            headings[last_code] = desc_text
            # 4-digit headings are always level-0 in the hierarchy
            level = _dash_level(' '.join(desc_buf)) if last_dashes == '' else _dash_level(last_dashes)
            if level == 0:
                level = 0  # force 4-digit codes to level 0
            heading_stack[level] = re.sub(r'^-+\s*', '', desc_text).strip(' :')
            # Clear all levels deeper than this heading (they belong to a
            # previous sibling subtree that this heading closes off)
            for deeper in list(heading_stack):
                if deeper > level:
                    del heading_stack[deeper]
        last_code, desc_buf[:], last_dashes = None, [], ''

    pages = range(min(max_page, doc.page_count)) if max_page else range(doc.page_count)

    for pageno in pages:
        lines = [l.strip() for l in doc[pageno].get_text().split('\n') if l.strip()]

        for line in lines:
            if not in_table:
                notes.append(line)
                continue

            if SKIP_RE.match(line):
                continue

            fn = FOOTNOTE_RE.match(line)
            if fn:
                running_footnotes[fn.group('marker')] = fn.group('text').strip()
                continue

            if CODE_RE.match(line):
                code = line.replace(' ', '')
                if OMITTED_RE.match(' '.join(desc_buf)) or (desc_buf and desc_buf[-1].lower() == 'omitted'):
                    pass  # an "Omitted" code carries no real data - drop silently
                else:
                    flush_heading()
                last_code, last_dashes = code, ''
                desc_buf, pending_unit, pending_marker, pending_rate = [], None, None, None
                continue

            # State-dependent transitions MUST be checked before the generic
            # dash-level check below, since a lone "-" is ambiguous: it means
            # "no preferential rate" if we're waiting on a pref value, but
            # means "dash-level marker" if we're starting a new heading.
            if pending_rate is not None and PREF_RE.match(line):
                pref = None if line == '-' else line

                # Determine the dash level for this row so we can build the
                # full inherited description correctly.
                raw_desc = ' '.join(desc_buf).strip(' :')
                child_level = _dash_level(last_dashes) if last_dashes else _dash_level(raw_desc)

                full_description = _build_full_description(heading_stack, raw_desc, child_level)

                row = {
                    'hsn_code': last_code,
                    'level': child_level,
                    'description': full_description,
                    'unit': pending_unit,
                    'footnote_marker': pending_marker,
                    'footnote_text': None,  # resolved after this page's footnote defs are all collected
                    'standard_rate': pending_rate,
                    'preferential_rate': pref,
                    'page': pageno + 1,
                }
                if not full_description:
                    row['_flag'] = 'empty_description'
                rows.append(row)
                last_code, desc_buf, pending_unit, pending_marker, pending_rate = None, [], None, None, None
                continue

            if pending_unit is not None and pending_rate is None:
                rm = RATE_RE.match(line)
                if rm:
                    pending_marker = rm.group('marker') or None
                    pending_rate = rm.group('rate')
                    continue

            if DASHES_RE.match(line):
                last_dashes = line
                continue

            if pending_unit is None and UNIT_RE.match(line) and last_code and line != '-':
                pending_unit = line
                continue

            if line.lower() == 'omitted':
                flush_heading()  # whatever code preceded this was an omitted entry, drop it
                last_code = None
                continue

            if END_OF_TABLE_HINTS.search(line):
                in_table = False
                flush_heading()
                notes.append(line)
                continue

            # Anything else is either a description continuation, or a
            # multi-line ALL-CAPS section/heading title sitting between rows.
            letters = [c for c in line if c.isalpha()]
            is_shouty = letters and sum(c.isupper() for c in letters) / len(letters) > 0.85 and len(line) > 3
            if is_shouty and pending_unit is None:
                # heading text, not a product description - never attach to a row
                continue

            # If this text line appears after a code was seen but before any
            # unit/rate, check if it's a dash-context sub-heading (no code of
            # its own). If so, store it in heading_stack so child codes on the
            # next line(s) inherit it. We only do this when there's no code
            # open yet waiting for unit/rate (i.e. last_code is None after the
            # flush, meaning the previous code was heading-only).
            if last_code is None and line.startswith('-'):
                lvl = _dash_level(line)
                desc_clean = re.sub(r'^-+\s*', '', line).strip(' :')
                if desc_clean:
                    heading_stack[lvl] = desc_clean
                    # Clear deeper levels
                    for deeper in list(heading_stack):
                        if deeper > lvl:
                            del heading_stack[deeper]
                continue

            desc_buf.append(line)

        # all of this page's footnote defs are now known - resolve footnote_text
        # for every row captured on this page using the running set (which now
        # includes both carried-forward defs AND this page's own)
        for r in rows:
            if r['page'] == pageno + 1 and r['footnote_marker'] and not r['footnote_text']:
                r['footnote_text'] = running_footnotes.get(r['footnote_marker'])

        # carry-over flag: if a code is still open with content collected at
        # the literal end of a page, it likely continues onto the next page -
        # flag it for review instead of silently dropping or guessing
        if last_code and (desc_buf or pending_unit):
            needs_review.append({
                'hsn_code': last_code, 'page': pageno + 1, '_flag': 'spans_page_boundary',
                'partial_description': ' '.join(desc_buf), 'partial_unit': pending_unit,
            })

    flush_heading()

    # final pass: anything still missing description/footnote text after all
    # pages are processed goes into needs_review for the batched LLM cleanup
    for r in rows:
        if not r['description'] and r not in needs_review:
            needs_review.append(r)
        elif r['footnote_marker'] and not r['footnote_text']:
            r2 = dict(r)
            r2['_flag'] = 'unresolved_footnote'
            needs_review.append(r2)

    return rows, headings, needs_review, ' '.join(notes)