"""
test_followups.py — exercises the conversational follow-up fix properly.

The built-in REPL at the bottom of rag_search.py calls answer_question(q)
fresh for every line you type, which builds a brand-new ONE-message list
each time — there is no conversation history between lines, so follow-up
resolution (which depends on scanning earlier messages) can never kick in
no matter how correct the logic is. This script instead keeps a running
`messages` list and appends each turn to it before calling
chat_with_context(), the way app.py's real /api/chat/mongo route does
when the frontend sends the full conversation each time.

Usage:
    python test_followups.py
"""
from rag_search import chat_with_context

CASES = [
    # (label, conversation so far, expected retrieval_method substring)
    (
        "direct hsn lookup",
        ["2923 20 90"],
        "direct_lookup:hsn_code",
    ),
    (
        "vague followup after hsn lookup",
        ["2923 20 90", "what is description"],
        "hsn_code_followup",
    ),
    (
        "another vague followup, same code",
        ["2923 20 90", "what is description", "give me more info on that"],
        "hsn_code_followup",
    ),
    (
        "chapter lookup",
        ["chapter 39"],
        "direct_lookup:chapter_number",
    ),
    (
        "real new question should NOT reuse stale chapter",
        ["chapter 39", "what's the GST exemption for fertilizers"],
        None,  # expect this NOT to be a chapter_number_followup
    ),

    # ── §5.1/5.3 regression cases: name-search precision fixes ──────────
    # "mango seed" vs. official description "Mango kernel" — only "mango"
    # ever matches, so this must resolve via the single-word tier rather
    # than falling through to unscoped vector search.
    (
        "name search: colloquial synonym (mango seed / Mango kernel)",
        ["give me import duty on mango seed"],
        "name_search:hsn_code",
    ),
    (
        "name search: multi-word raw-PDF-text match (neem seeds)",
        ["give me import duty on neem seeds"],
        "name_search:hsn_code",
    ),
    (
        "name search: bare product name shouldn't return a derived product first",
        ["poppy straw"],
        "name_search:hsn_code",
    ),

    # ── §5.4 regression case: loose tier must not short-circuit ─────────
    # "sugar cane" sharing only the word "sugar" with "Sugars, chemically
    # pure..." must NOT come back as a same-confidence answer — either a
    # better hybrid-RAG hit is found, or the retrieval_method explicitly
    # marks it as a LAST-RESORT loose match with a caveat attached.
    (
        "name search: loose single-word match must not short-circuit",
        ["give me import duty on sugar cane"],
        None,  # checked specially below, not a plain substring match
    ),

    # ── Newly found bug: "partial" tier (2+ shared words) is JUST as
    # unreliable as "loose" and was short-circuiting too. Real examples:
    # "Durries of man-made fibres" matched 7 unrelated carpet-fibre rows
    # via "man-made"/"fibres" alone, missing the real Chapter 57 row
    # (HSN 5705 00 22); "Vaccines for veterinary medicine" matched an
    # unrelated "Anaesthetic agents... veterinary medicine" row via
    # "veterinary"/"medicine" alone, missing HSN 3002 42 00. Both are now
    # deferred the same way "loose" is.
    (
        "name search: partial 2-word match must not mask the real row (durries)",
        ["give me import duty on Durries of man-made fibres"],
        None,  # checked specially below
    ),
    (
        "name search: partial 2-word match must not mask the real row (vaccines)",
        ["give me import duty on Vaccines for veterinary medicine"],
        None,  # checked specially below
    ),

    # ── §5.2 regression case: duty-rate trend feature ────────────────────
    (
        "trend lookup: duty rate history for an HSN code",
        ["what is the duty rate trend for HSN 2939 11 00"],
        "trend_lookup:hsn_code",
    ),
    (
        "trend lookup: conversational followup referencing prior HSN code",
        ["2923 20 90", "how has the duty rate on that changed over time"],
        "trend_lookup:hsn_code",
    ),

    # ── Regression: name-based comparison ("compare X and Y") ───────────
    # Previously fell through to the bare-name fallback (treating the
    # whole "compare polyethylene and pvc" string as ONE garbled search
    # term) and, on any short follow-up mentioning "duty"/"rate", was
    # silently swallowed by the single-entity vague-followup path, which
    # only remembers the last individually-mentioned HSN code (here: PVC)
    # and repeats that same single-code answer regardless of rewording.
    (
        "name comparison: two product names in one message",
        ["what is import duty on polyethylene", "3904 10 10", "compare polyethylene and pvc"],
        None,  # checked specially: must be a genuine multi-row comparison
    ),
    (
        "name comparison: vague followup must reuse both prior terms, not one stale code",
        ["what is import duty on polyethylene", "3904 10 10",
         "compare polyethylene and pvc", "compare duty rate"],
        None,
    ),
    (
        "name comparison: reworded vague followup ('make table') must not repeat stale single code",
        ["what is import duty on polyethylene", "3904 10 10",
         "compare polyethylene and pvc", "compare polyethylene and pvc duty rate make table"],
        None,
    ),
]

def _is_genuine_multi_row_comparison(method: str) -> bool:
    """
    True only if this resolved via the templated multi-code table path
    with 2+ docs — NOT the old single-entity "_followup" bug, which
    would show up as e.g. "direct_lookup:hsn_code_followup" (exactly
    one doc, silently reusing a stale single code and ignoring the
    comparison).
    """
    return "hsn_code_multi" in method or "hsn_code_mixed" in method

# Cases handled specially in run() because a plain substring check on
# retrieval_method isn't enough to express what "correct" means for them.
def _low_confidence_deferred_correctly(method: str) -> bool:
    """
    True if a query that's expected to hit only the "loose" or "partial"
    match tier was handled correctly: either something MORE confident
    (all_words/phrase/exact) answered it, or the hybrid RAG fallback
    found a real chapter match, or -- only as an actual last resort --
    it fell back to the stashed loose/partial guess AND that fallback is
    explicitly labeled LAST-RESORT (never a bare, un-flagged
    match=loose/partial, which would mean it short-circuited again).
    """
    return (
        ("name_search:hsn_code" in method and "match=loose" not in method and "match=partial" not in method)
        or "tariff_schedule_vector_fallback" in method
        or ("LAST-RESORT" in method and ("match=loose" in method or "match=partial" in method))
    )


SPECIAL_CASES = {
    "name search: loose single-word match must not short-circuit": _low_confidence_deferred_correctly,
    "name search: partial 2-word match must not mask the real row (durries)": _low_confidence_deferred_correctly,
    "name search: partial 2-word match must not mask the real row (vaccines)": _low_confidence_deferred_correctly,
    "name comparison: two product names in one message": _is_genuine_multi_row_comparison,
    "name comparison: vague followup must reuse both prior terms, not one stale code": _is_genuine_multi_row_comparison,
    "name comparison: reworded vague followup ('make table') must not repeat stale single code": _is_genuine_multi_row_comparison,
}

def run():
    passed, failed, errored = 0, 0, 0
    for label, turns, expect in CASES:
        messages = [{"role": "user", "content": t} for t in turns]
        try:
            result = chat_with_context(messages)
        except Exception as e:
            # Don't let one transient failure (e.g. Gemini 503 "high demand")
            # take down the rest of the suite -- log it as a failure for
            # THIS case and keep going, so a single flaky API call doesn't
            # hide results for every case after it.
            errored += 1
            print(f"⚠️  ERROR — {label}")
            print(f"   turns: {turns}")
            print(f"   exception: {type(e).__name__}: {e}")
            print()
            continue

        method = result.get("retrieval_method", "")
        if label in SPECIAL_CASES:
            ok = SPECIAL_CASES[label](method)
        elif expect:
            ok = expect in method
        else:
            ok = "chapter_number_followup" not in method
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{status} — {label}")
        print(f"   turns: {turns}")
        print(f"   retrieval_method: {method}")
        print(f"   response: {result.get('response', '')[:200]}")
        print()

    print(f"\n{passed} passed, {failed} failed, {errored} errored (transient/exception, re-run these)")

if __name__ == "__main__":
    run()