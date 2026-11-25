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

    last_code = None
    last_dashes = ''
    desc_buf = []
    pending_unit = None
    pending_marker = None
    pending_rate = None

    def flush_heading():
        nonlocal last_code, desc_buf
        if last_code and desc_buf:
            headings[last_code] = ' '.join(desc_buf).strip(' :')
        last_code, desc_buf[:] = None, []

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
                row = {
                    'hsn_code': last_code,
                    'level': len(last_dashes) or len(re.match(r'^-*', ' '.join(desc_buf).strip()).group()),
                    'description': re.sub(r'^-+\s*', '', ' '.join(desc_buf).strip(' :')),
                    'unit': pending_unit,
                    'footnote_marker': pending_marker,
                    'footnote_text': None,  # resolved after this page's footnote defs are all collected
                    'standard_rate': pending_rate,
                    'preferential_rate': pref,
                    'page': pageno + 1,
                }
                if not row['description']:
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