import os
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pymongo import MongoClient
from dotenv import load_dotenv

from rag_search import chat_with_context, call_gemini

load_dotenv()

app = Flask(__name__, static_folder="static")

# ── CORS ──────────────────────────────────────────────────────────────────────
# Wide-open CORS (CORS(app) with no args) lets ANY website's frontend JS call
# this API using a visiting user's browser session. For a demo running only
# on your own machine this doesn't matter, but the moment this is reachable
# from the internet it should be locked down. Set ALLOWED_ORIGINS in .env as
# a comma-separated list, e.g. "https://regulai.example.com". Falls back to
# "*" (open) only when unset, so local dev still works without extra config.
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
CORS(app, origins=_allowed_origins.split(",") if _allowed_origins != "*" else "*")

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Without this, anyone with the URL can hammer /api/chat/mongo — each call
# costs a Gemini API request and an Atlas vector search, so this is a real
# cost/availability risk, not just a theoretical one. Defaults are intentionally
# generous (this is a demo app, not a hardened SaaS) — tune via env if needed.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[os.getenv("RATE_LIMIT_DEFAULT", "60 per hour")],
    storage_uri="memory://",  # fine for a single-process demo; swap for Redis if you scale to multiple workers
)

# ── API Keys ──────────────────────────────────────────────────────────────────
# GEMINI_API_KEY and the model name now live in rag_search.py — call_gemini()
# (imported above) reads them internally via the native google-genai SDK, so
# there's nothing for app.py to hold or pass on its own behalf anymore.

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI  = os.getenv("MONGO_URI", "YOUR_MONGODB_")
DB_NAME    = "regulai"
COLLECTION = "notifications"

_mongo_client = None
_mongo_col    = None

def get_mongo_col():
    """Lazy singleton — connects once, reuses forever."""
    global _mongo_client, _mongo_col
    if _mongo_col is None:
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)
        _mongo_col    = _mongo_client[DB_NAME][COLLECTION]
    return _mongo_col

# ── Chat constants ────────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS = 24_000
CHUNK_SIZE        = 800
TOP_K_CHUNKS      = 20
MAX_HISTORY_TURNS = 6


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_relevant_chunks(query, text):
    if len(text) <= MAX_CONTEXT_CHARS:
        return text
    chunks = [(i, text[i:i+CHUNK_SIZE]) for i in range(0, len(text), CHUNK_SIZE)]
    stop   = {
        'the','a','an','is','are','was','were','what','which','who','how',
        'when','where','why','do','does','did','can','could','would','should',
        'will','be','been','have','has','had','this','that','i','me','my',
        'you','your','it','in','on','at','to','for','of','and','or','show',
        'tell','give','list','find','explain','please','all','any','about'
    }
    query_words = set(re.findall(r'\w+', query.lower())) - stop
    scored = sorted(
        [(sum(1 for w in query_words if w in c.lower()), i, c) for i, c in chunks],
        key=lambda x: x[0], reverse=True
    )
    top = sorted(scored[:TOP_K_CHUNKS], key=lambda x: x[1])
    return '\n\n'.join(c for _, _, c in top)[:MAX_CONTEXT_CHARS]


def trim_history(messages):
    if len(messages) <= MAX_HISTORY_TURNS * 2:
        return messages
    return messages[-(MAX_HISTORY_TURNS * 2):]


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error: " + str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── 1. Original chat (PDF text passed from frontend) ─────────────────────────
@app.route("/api/chat", methods=["POST"])
@limiter.limit(os.getenv("RATE_LIMIT_CHAT", "20 per hour"))
def chat():
    try:
        data     = request.get_json(force=True, silent=True) or {}
        messages = data.get("messages", [])
        pdf_text = data.get("pdfText", "")

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        latest_question = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        relevant_text = get_relevant_chunks(latest_question, pdf_text)
        return _call_gemini(messages, relevant_text)

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 200


# ── 2. Chat powered by MongoDB Atlas Vector Search (RAG) ─────────────────────
@app.route("/api/chat/mongo", methods=["POST"])
@limiter.limit(os.getenv("RATE_LIMIT_CHAT", "20 per hour"))
def chat_mongo():
    """
    RAG chat endpoint. For the latest user question:
      1. Tries a direct notification-number lookup against 'notifications'
         (e.g. "notification 50/2017") — bypasses vector search if found.
      2. Otherwise runs $vectorSearch against 'chunks' (built by
         embed_pipeline.py). By default this excludes "Tariff Schedule"
         docs unless the question itself looks like a chapter/HSN query
         (e.g. "what's in chapter 39"), since most real questions are
         about notifications, not static tariff chapter listings.
    Sends the retrieved context + conversation history to Gemini.

    Body: { "messages": [...], "category": "optional — overrides auto-filtering above" }
    """
    try:
        data     = request.get_json(force=True, silent=True) or {}
        messages = data.get("messages", [])
        category = data.get("category") or None

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        result = chat_with_context(messages, top_k=5, category=category)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 200


# ── 3. Search notifications in MongoDB ───────────────────────────────────────
@app.route("/api/search", methods=["GET"])
def search_notifications():
    """
    GET /api/search?q=polymer&category=Import+Policy&limit=10
    Returns matching notifications (metadata only, no full_text).
    """
    try:
        q        = request.args.get("q", "").strip()
        category = request.args.get("category", "").strip()
        limit    = int(request.args.get("limit", 10))

        docs = search_mongo(q, category=category or None, limit=limit)

        results = [{
            "notification_id": d.get("notification_id"),
            "notification_no": d.get("notification_no"),
            "title":           d.get("title"),
            "date":            d.get("date"),
            "category":        d.get("category"),
            "keywords":        d.get("keywords", []),
            "pdf_url":         d.get("pdf_url"),
        } for d in docs]

        return jsonify({"count": len(results), "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 4. Get a single notification by ID ───────────────────────────────────────
@app.route("/api/notification/<notification_id>", methods=["GET"])
def get_notification(notification_id):
    """Returns full document including full_text."""
    try:
        col = get_mongo_col()
        doc = col.find_one({"notification_id": notification_id}, {"_id": 0})
        if not doc:
            return jsonify({"error": "Not found"}), 404
        return jsonify(doc)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 5. Stats / health check ───────────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def stats():
    """Returns quick DB stats — useful for dashboard."""
    try:
        col   = get_mongo_col()
        total = col.count_documents({})

        pipeline = [{"$group": {"_id": "$category", "count": {"$sum": 1}}}]
        by_cat   = {d["_id"]: d["count"] for d in col.aggregate(pipeline)}

        return jsonify({
            "total_notifications": total,
            "by_category":         by_cat,
            "db":                  DB_NAME,
            "collection":          COLLECTION,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Internal: MongoDB search
# ─────────────────────────────────────────────────────────────────────────────
def search_mongo(query: str, category: str = None, limit: int = 5) -> list:
    """
    Search MongoDB using:
    1. Atlas full-text search (if text index exists)
    2. Fallback: regex on title field

    Uses re.compile() rather than the dict-style {"$regex": ..., "$options": "i"}
    operator — confirmed by direct testing (see rag_search.py) that the dict
    form silently matches zero documents on this PyMongo/Atlas setup, while
    re.compile() works correctly against the same fields.
    """
    col    = get_mongo_col()
    filter_ = {}

    if category:
        filter_["category"] = category

    if query:
        # Try full-text search first
        try:
            cursor = col.find(
                {**filter_, "$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "_id": 0}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            docs = list(cursor)
            if docs:
                return docs
        except Exception:
            pass  # text index might not exist yet

        # Fallback: regex on title + keywords
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        filter_["$or"] = [
            {"title":    pattern},
            {"keywords": pattern},
            {"notification_no": pattern},
        ]

    cursor = col.find(filter_, {"_id": 0}).limit(limit)
    return list(cursor)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: Gemini call
# ─────────────────────────────────────────────────────────────────────────────
def _call_gemini(messages: list, context: str):
    # `context` has already been passed through get_relevant_chunks() by the
    # caller (chat()) — re-running it here was a no-op (the result is already
    # under MAX_CONTEXT_CHARS) but misleading to read, so use it directly.
    relevant_text = context

    system_prompt = f"""You are RegulAI, an expert AI assistant specialized in Indian trade law, regulatory compliance, customs duties, import/export regulations, and government notifications.

You are analyzing the following regulatory notifications and documents:

--- REGULATORY DOCUMENTS ---
{relevant_text}
--- END DOCUMENTS ---

Your role:
- Answer questions about regulatory notifications, trade laws, customs duties, import/export restrictions.
- Always cite the specific Notification Number and page when referencing content.
  Example: "As per Notification No. 12/2023-Customs (Page 3)..."
- Identify amendments, superseded notifications, and regulatory relationships.
- Highlight compliance requirements clearly.
- When listing regulations, use structured formatting with bullet points.
- Flag any prohibitions, restrictions, or exemptions clearly.
- If asked to compare notifications, do so in a structured table format.
- If information is not in the documents, clearly state that.
- If the user states a fact, code, description, or rate themselves (rather than it appearing in the documents above), do not repeat it back as if it were confirmed by the documents. Acknowledge it as something they provided and say plainly whether or not you can verify it against the documents shown above.

Format your responses professionally for compliance teams and legal departments.
Use **bold** for notification numbers, key terms, and important compliance points.
"""

    gemini_messages = [
        {"role": ("user" if msg["role"] == "user" else "assistant"), "content": msg["content"]}
        for msg in trim_history(messages)
    ]

    try:
        result = call_gemini(messages=gemini_messages, system_prompt=system_prompt, max_tokens=1000)
        text = result["text"]
        finish_reason = result["finish_reason"]
        if finish_reason and finish_reason not in ("STOP", "stop"):
            print(f"⚠ Gemini response did not finish cleanly — finish_reason={finish_reason}")
            text += (
                f"\n\n*(Note: this response was cut short by the model — finish_reason: "
                f"{finish_reason}. Try asking a shorter or more specific question.)*"
            )
        return jsonify({"response": text})

    except RuntimeError as e:
        # call_gemini raises this if GEMINI_API_KEY is missing/unset.
        return jsonify({"error": str(e)}), 200
    except Exception as e:
        # google-genai surfaces API errors (rate limits, bad key shape, etc.)
        # as its own exception types rather than HTTP status codes, so this
        # catches anything else rather than checking resp.status_code the
        # way the old REST-based version did.
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            return jsonify({"error": "Rate limit hit. Please wait a moment and try again."}), 200
        return jsonify({"error": f"Gemini API error: {msg}"}), 200


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # FLASK_ENV controls both the debugger and which server runs. The Flask
    # debugger (debug=True) is convenient locally but if it's ever reachable
    # from outside your machine it allows arbitrary code execution through
    # its interactive traceback page — so it should never be on by default.
    # Set FLASK_ENV=development in .env to get the old debug-reloader behavior
    # back for local work; anything else (including unset) runs production-safe.
    is_dev = os.getenv("FLASK_ENV", "production").lower() == "development"
    port   = int(os.getenv("PORT", 5000))

    print(f"\n RegulAI running at http://localhost:{port}  ({'development' if is_dev else 'production'} mode)")
    print(f"   MongoDB: {DB_NAME}.{COLLECTION}")
    print("   Endpoints:")
    print("     POST /api/chat         — chat with uploaded PDF")
    print("     POST /api/chat/mongo   — chat with vector-search RAG over MongoDB")
    print("     GET  /api/search       — search notifications")
    print("     GET  /api/notification/<id> — get single doc")
    print("     GET  /api/stats        — DB stats\n")

    if is_dev:
        app.run(debug=True, port=port)
    else:
        # Flask's built-in server (app.run) is single-threaded and explicitly
        # documented as unsuitable for production — it can't handle concurrent
        # requests well and has no protection against slow/malicious clients.
        # Waitress is a pure-Python WSGI server (no compiled deps, works fine
        # on Windows, unlike gunicorn) — a low-effort swap that's actually
        # meant to be exposed.
        from waitress import serve
        serve(app, host="0.0.0.0", port=port)