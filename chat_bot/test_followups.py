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
]

def run():
    passed, failed = 0, 0
    for label, turns, expect in CASES:
        messages = [{"role": "user", "content": t} for t in turns]
        result = chat_with_context(messages)
        method = result.get("retrieval_method", "")
        ok = (expect in method) if expect else ("chapter_number_followup" not in method)
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

    print(f"\n{passed} passed, {failed} failed")

if __name__ == "__main__":
    run()