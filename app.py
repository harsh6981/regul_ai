import sys
# Configure stdout/stderr to UTF-8 encoding on startup to prevent Windows console / redirect charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import re
import uuid
import io
import threading
import contextlib
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pymongo import MongoClient
from dotenv import load_dotenv

from rag_search import chat_with_context, call_gemini
from auth import create_user, verify_user, issue_token, require_auth

# Pre-import the scrape-pipeline modules HERE, at real startup, against the
# real console stdout — NOT lazily inside run_full_pipeline(). project.py has
# a one-time top-level check (`sys.stdout.encoding.lower()`, same pattern as
# this file's own lines 3-6) that runs the first time it's imported. If that
# first import happened lazily inside run_full_pipeline()'s
# `contextlib.redirect_stdout(io.StringIO())`, sys.stdout would be a StringIO
# at that moment — and io.StringIO().encoding is None, not a string, so
# `.lower()` on it blows up with "'NoneType' object has no attribute 'lower'".
# Importing eagerly here means that check runs once, safely, before any
# redirect ever happens; every later `import project` (e.g. in
# run_full_pipeline) just reuses this cached module and re-runs none of its
# top-level code.
import project
import mongo_pipeline
import embed_pipeline

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


CHATS_COLLECTION = "chats"
_chats_col = None

def get_chats_col():
    """
    Lazy singleton for the per-user chat-thread collection. Reuses the same
    MongoClient get_mongo_col() already opened rather than making a second
    connection.
    """
    global _chats_col
    if _chats_col is None:
        col = get_mongo_col().database[CHATS_COLLECTION]
        col.create_index([("user_email", 1), ("updated_at", -1)])
        col.create_index("chat_id", unique=True)
        _chats_col = col
    return _chats_col

# ─────────────────────────────────────────────────────────────────────────────
# Scrape pipeline — shared by the manual "New Notifications" button AND the
# scheduled background job below, so both paths run the exact same code and
# can never run concurrently (a scheduled run won't stomp on a manual click
# mid-crawl, or vice versa).
# ─────────────────────────────────────────────────────────────────────────────
_scrape_lock  = threading.Lock()
_scrape_state = {
    "running":    False,
    "status":     "idle",     # idle | running | complete | error
    "started_at": None,
    "finished_at": None,
    "trigger":    None,       # "manual" | "scheduled"
    "error":      None,
    "log":        "",
}


def run_full_pipeline(trigger: str = "manual"):
    """
    Runs project.main() -> mongo_pipeline.run_pipeline() ->
    embed_pipeline.run_pipeline() in order, capturing their stdout into
    _scrape_state["log"] so the frontend (or scheduler) can show progress
    without needing a real job queue. Safe to call from a request handler
    (in a background thread) or directly from APScheduler.
    """
    if not _scrape_lock.acquire(blocking=False):
        print(f"[scrape] Skipped {trigger} run — a scrape is already in progress.")
        return
    log = io.StringIO()
    _scrape_state.update(running=True, status="running", trigger=trigger,
                          started_at=datetime.utcnow().isoformat(),
                          finished_at=None, error=None, log="")
    try:
        with contextlib.redirect_stdout(log):
            import project
            project.main()

            import mongo_pipeline
            mongo_pipeline.run_pipeline()

            import embed_pipeline
            embed_pipeline.run_pipeline()

        _scrape_state.update(status="complete")
    except Exception as e:
        _scrape_state.update(status="error", error=str(e))
        print(f"[scrape] {trigger} run failed: {e}")
    finally:
        _scrape_state.update(
            running=False,
            finished_at=datetime.utcnow().isoformat(),
            log=log.getvalue()[-5000:],
        )
        _scrape_lock.release()


# ── Scheduled scraping ────────────────────────────────────────────────────────
# Config lives in Mongo (settings.scrape_schedule) so it can be changed from
# the UI at any time without touching .env or restarting the server. .env
# vars (SCRAPE_SCHEDULE_ENABLED etc.) are only the one-time default used the
# very first time the app boots and no settings doc exists yet.
#
#   enabled          bool
#   type             "cron" | "interval"
#   day_of_week      "mon", "mon,thu", "mon-fri", or "*" for every day  (cron)
#   hour             0-23                                              (cron)
#   minute           0-59                                              (cron)
#   interval_hours   e.g. 24                                        (interval)
DEFAULT_SCHEDULE = {
    "enabled":        os.getenv("SCRAPE_SCHEDULE_ENABLED", "false").lower() == "true",
    "type":           os.getenv("SCRAPE_SCHEDULE_TYPE", "cron").lower(),
    "day_of_week":    os.getenv("SCRAPE_CRON_DAY_OF_WEEK", "mon"),
    "hour":           int(os.getenv("SCRAPE_CRON_HOUR", "3")),
    "minute":         int(os.getenv("SCRAPE_CRON_MINUTE", "0")),
    "interval_hours": float(os.getenv("SCRAPE_INTERVAL_HOURS", "24")),
}


def get_settings_col():
    return get_mongo_col().database["settings"]


def load_schedule() -> dict:
    doc = get_settings_col().find_one({"_id": "scrape_schedule"})
    return {**DEFAULT_SCHEDULE, **(doc or {})}


def save_schedule(cfg: dict) -> dict:
    merged = {**load_schedule(), **cfg}
    merged["_id"] = "scrape_schedule"
    get_settings_col().replace_one({"_id": "scrape_schedule"}, merged, upsert=True)
    return merged


_scheduler = None  # BackgroundScheduler instance, created once at boot and reused


def apply_schedule(cfg: dict):
    """
    (Re)configures the live 'scrape_pipeline' job to match cfg — called at
    boot and again every time the settings UI saves changes, so a schedule
    edit takes effect immediately with no restart needed.
    """
    global _scheduler
    if _scheduler is None:
        from apscheduler.schedulers.background import BackgroundScheduler
        _scheduler = BackgroundScheduler(daemon=True)
        _scheduler.start()

    if _scheduler.get_job("scrape_pipeline"):
        _scheduler.remove_job("scrape_pipeline")

    if not cfg.get("enabled"):
        print("[scrape] Scheduled scraping OFF")
        return

    job_kwargs = dict(
        id="scrape_pipeline",
        max_instances=1,   # never overlap a slow run with the next tick
        coalesce=True,     # if the process was asleep through a tick, run once, not N times
    )

    if cfg.get("type") == "interval":
        _scheduler.add_job(
            lambda: run_full_pipeline(trigger="scheduled"),
            "interval",
            hours=float(cfg["interval_hours"]),
            **job_kwargs,
        )
        print(f"[scrape] Scheduled scraping ON — every {cfg['interval_hours']}h")
    else:
        from apscheduler.triggers.cron import CronTrigger
        trigger = CronTrigger(day_of_week=cfg["day_of_week"], hour=cfg["hour"], minute=cfg["minute"])
        _scheduler.add_job(lambda: run_full_pipeline(trigger="scheduled"), trigger, **job_kwargs)
        print(f"[scrape] Scheduled scraping ON — day_of_week='{cfg['day_of_week']}' "
              f"at {int(cfg['hour']):02d}:{int(cfg['minute']):02d}")


def start_scheduler():
    """Called once at app boot — loads the saved (or default) schedule and applies it."""
    apply_schedule(load_schedule())
    return _scheduler



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


# ── Auth: signup / login / session restore ────────────────────────────────────
@app.route("/api/auth/signup", methods=["POST"])
@limiter.limit("10 per hour")
def signup():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        user  = create_user(data.get("email", ""), data.get("password", ""))
        token = issue_token(user["email"])
        return jsonify({"token": token, "email": user["email"]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/auth/login", methods=["POST"])
@limiter.limit("20 per hour")
def login():
    try:
        data  = request.get_json(force=True, silent=True) or {}
        user  = verify_user(data.get("email", ""), data.get("password", ""))
        token = issue_token(user["email"])
        return jsonify({"token": token, "email": user["email"]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/auth/me", methods=["GET"])
@require_auth
def me():
    """
    Session-restore check. Frontend calls this on every page load with
    whatever token it has in localStorage — a 200 here means the token's
    still valid and the user stays logged in as request.user_email
    without re-entering a password; a 401 means show the login page.
    """
    return jsonify({"email": request.user_email})
@app.route('/api/auth/logout', methods=['POST'])
@require_auth  # Optional: verify they had a valid token first
def logout():
    """
    Stateless endpoint for logout. Since you're using JWTs (not 
    server-side sessions), there's nothing to clear server-side — 
    just tell the client to clear localStorage.
    """
    return {"status": "logged out"}, 200


# ── Chat threads: list / create / load / save / delete ────────────────────────
@app.route("/api/chats", methods=["GET", "POST"])
@require_auth
def chats_collection_route():
    col = get_chats_col()

    if request.method == "POST":
        # The very first chat a user ever creates becomes their pinned,
        # non-deletable "RegulAI Database" thread; every one after that is
        # a normal, closeable "New Chat" thread.
        is_first = col.count_documents({"user_email": request.user_email}) == 0
        now = datetime.utcnow().isoformat()
        doc = {
            "chat_id":    uuid.uuid4().hex,
            "user_email": request.user_email,
            "title":      "RegulAI Database" if is_first else "New Chat",
            "pinned":     is_first,
            "messages":   [],
            "created_at": now,
            "updated_at": now,
        }
        col.insert_one(doc)
        doc.pop("_id", None)
        return jsonify(doc)

    # GET — list this user's chats, newest first, pinned always on top.
    # Message bodies are left out here on purpose (can get large); the
    # frontend fetches them individually via GET /api/chats/<id> only when
    # that thread is actually opened.
    docs = list(col.find(
        {"user_email": request.user_email},
        {"_id": 0, "messages": 0}
    ).sort([("pinned", -1), ("updated_at", -1)]))
    return jsonify({"chats": docs})


@app.route("/api/chats/<chat_id>", methods=["GET", "PUT", "DELETE"])
@require_auth
def single_chat_route(chat_id):
    col = get_chats_col()
    doc = col.find_one({"chat_id": chat_id, "user_email": request.user_email})
    if not doc:
        return jsonify({"error": "Chat not found"}), 404

    if request.method == "GET":
        doc.pop("_id", None)
        return jsonify(doc)

    if request.method == "DELETE":
        if doc.get("pinned"):
            return jsonify({"error": "The default database chat can't be deleted"}), 400
        col.delete_one({"chat_id": chat_id})
        return jsonify({"status": "deleted"})

    # PUT — called after every turn to persist the full message list, so a
    # refresh (or logging in from another device) picks up right where the
    # conversation left off.
    data = request.get_json(force=True, silent=True) or {}
    messages = data.get("messages")
    if messages is None:
        return jsonify({"error": "messages field is required"}), 400

    update = {"messages": messages, "updated_at": datetime.utcnow().isoformat()}
    # Auto-title once from the first user message, rather than leaving every
    # thread named "New Chat" forever.
    if doc.get("title") in (None, "New Chat"):
        first_user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), None)
        if first_user_msg:
            update["title"] = (first_user_msg[:42] + "…") if len(first_user_msg) > 42 else first_user_msg

    col.update_one({"chat_id": chat_id}, {"$set": update})
    return jsonify({"status": "saved", "title": update.get("title", doc.get("title"))})


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

    Body: { "messages": [...], "category": "optional — overrides auto-filtering above",
            "attachedPdfText": "optional — text of a PDF attached client-side to this
             chat via the paperclip button. NEVER folded into the message that's used
             for Mongo lookups/routing (extract_*_number(), name-search, etc. all build
             re.compile()/regex queries straight out of message text — feeding those a
             multi-thousand-character blob produces a query so large Atlas's proxy
             rejects it outright: 'Unable to parse body section as bson: bufio: buffer
             full'). Instead, retrieval runs on the plain question only, and the
             attachment is blended in afterward via a second, Mongo-free Gemini call.
             Never persisted (the chat thread saved via PUT /api/chats/<id> keeps the
             plain question text, not this attachment). }
    """
    try:
        data     = request.get_json(force=True, silent=True) or {}
        messages = data.get("messages", [])
        category = data.get("category") or None
        attached_pdf_text = (data.get("attachedPdfText") or "").strip()[:10_000]

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        # Retrieval/routing always runs on the messages exactly as sent —
        # the attachment (if any) never touches this call.
        result = chat_with_context(messages, top_k=5, category=category)

        if attached_pdf_text:
            original_question = messages[-1].get("content", "") if messages else ""
            blend_prompt = (
                f"Original question: {original_question}\n\n"
                f"--- Answer already grounded in the CBIC regulatory database ---\n"
                f"{result.get('response', '')}\n\n"
                f"--- Additional document attached by the user for this question "
                f"(NOT part of the CBIC database) ---\n"
                f"{attached_pdf_text}\n\n"
                f"Using both the database-grounded answer and the attached document, "
                f"write one combined, well-organized answer to the original question. "
                f"If they cover different things, make clear which parts come from the "
                f"CBIC database versus the attached document."
            )
            try:
                blended = call_gemini(
                    messages=[{"role": "user", "content": blend_prompt}],
                    system_prompt=(
                        "You are RegulAI, a regulatory intelligence assistant for Indian "
                        "Customs. Combine the two sources of information given into a "
                        "single clear, well-organized answer."
                    ),
                    max_tokens=1500,
                    temperature=0.3,
                    thinking_budget=0,
                )
                if blended.get("text"):
                    result["response"] = blended["text"]
            except Exception as blend_err:
                # Attachment blending is a nice-to-have on top of an already-good
                # DB-grounded answer — if it fails for any reason (rate limit,
                # provider hiccup), fall back to that answer rather than erroring
                # out the whole request.
                print(f"[chat_mongo] attachment blend failed, using DB-only answer: {blend_err}")

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


# ── 5a. Manual PDF upload ────────────────────────────────────────────────────
PDF_FOLDER = os.getenv("PDF_STORAGE_PATH", "CBIC_ALL_PDFS")
PDF_UPLOAD_FOLDER = os.path.join(PDF_FOLDER, "manual_uploads")
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)


@app.route("/api/pdf/upload", methods=["POST"])
@require_auth
@limiter.limit("30 per hour")
def upload_pdf():
    """
    Manual PDF ingestion, always treated as a notification/circular
    (extract text -> embed -> chat-searchable), same shape as a scraped
    doc but tagged source="manual_upload". Done inline so the PDF is
    searchable in chat immediately after this call returns — no separate
    pipeline run needed.

    multipart/form-data:
      file             - the PDF (required)
      title            - optional, defaults to the filename
      notification_no  - optional, defaults to "N/A"
      category         - optional, defaults to "Notification"
      date             - optional, defaults to "N/A"
    """
    import uuid
    import datetime as dt
    from werkzeug.utils import secure_filename
    from pymongo import UpdateOne

    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file' in request"}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    try:
        safe_name = secure_filename(file.filename)
        notification_id = f"manual-{uuid.uuid4().hex[:8]}-{safe_name.rsplit('.', 1)[0]}"
        save_path = os.path.join(PDF_UPLOAD_FOLDER, f"{notification_id}.pdf")
        file.save(save_path)

        from mongo_pipeline import extract_pdf_text
        full_text = extract_pdf_text(save_path)
        if not full_text:
            return jsonify({
                "error": "Couldn't extract any text from this PDF "
                         "(scanned/image-only PDFs aren't supported yet)"
            }), 400

        now = dt.datetime.utcnow().isoformat()
        doc = {
            "notification_id":  notification_id,
            "notification_no":  request.form.get("notification_no", "N/A"),
            "title":            request.form.get("title") or safe_name.rsplit(".", 1)[0],
            "date":             request.form.get("date", "N/A"),
            "authority":        "CBIC",
            "category":         request.form.get("category", "Notification"),
            "keywords":         [],
            "summary":          "",
            "full_text":        full_text,
            "pdf_url":          "",
            "file_location":    save_path,
            "related_notifications": [],
            "embedding":        [],
            "source":           "manual_upload",
            "uploaded_by":      request.user_email,
            "uploaded_at":      now,
            "indexed_at":       now,
        }

        notif_col = get_mongo_col()
        notif_col.update_one({"notification_id": notification_id}, {"$set": doc}, upsert=True)

        # Chunk + embed immediately (same logic embed_pipeline.py runs in bulk)
        from embed_pipeline import chunk_text, embed_passages
        pieces = chunk_text(full_text)
        chunks_indexed = 0
        if pieces:
            vectors = embed_passages(pieces)
            client     = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)
            chunks_col = client[DB_NAME]["chunks"]
            ops = []
            for idx, (piece, vec) in enumerate(zip(pieces, vectors)):
                chunk_doc = {
                    "chunk_id":        f"{notification_id}__{idx}",
                    "notification_id": notification_id,
                    "chunk_index":     idx,
                    "text":            piece,
                    "embedding":       vec,
                    "title":           doc["title"],
                    "category":        doc["category"],
                    "notification_no": doc["notification_no"],
                    "date":            doc["date"],
                    "pdf_url":         "",
                    "indexed_at":      now,
                }
                ops.append(UpdateOne({"chunk_id": chunk_doc["chunk_id"]}, {"$set": chunk_doc}, upsert=True))
            chunks_col.bulk_write(ops, ordered=False)
            client.close()
            chunks_indexed = len(pieces)

        return jsonify({
            "status":           "success",
            "notification_id":  notification_id,
            "title":            doc["title"],
            "chunks_indexed":   chunks_indexed,
            "message":          "PDF processed and is now searchable in chat.",
        })

    except Exception as e:
        return jsonify({"error": f"Upload processing failed: {e}"}), 500


# ── 5b. Manual scrape trigger ──────────────────────────────────────────────────
@app.route("/api/scrape/run", methods=["POST"])
@require_auth
@limiter.limit("2 per hour")
def run_scrape():
    """
    Kicks off the full ingest chain (crawl -> extract -> chunk/embed) in a
    background thread and returns immediately — a full crawl can take
    several minutes, too long to hold an HTTP request open. Poll
    GET /api/scrape/status for progress; it's the same status the
    scheduled job (see start_scheduler()) also writes to, so only one
    scrape can ever be in flight at a time regardless of who triggered it.
    """
    if _scrape_state["running"]:
        return jsonify({"status": "already_running", **_scrape_state}), 409
    threading.Thread(target=run_full_pipeline, kwargs={"trigger": "manual"}, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/scrape/status", methods=["GET"])
def scrape_status():
    """Poll this after POST /api/scrape/run (or just to see the last/next scheduled run)."""
    state = dict(_scrape_state)
    if _scheduler:
        job = _scheduler.get_job("scrape_pipeline")
        state["next_scheduled_run"] = job.next_run_time.isoformat() if job and job.next_run_time else None
    else:
        state["next_scheduled_run"] = None
    return jsonify(state)


@app.route("/api/scrape/schedule", methods=["GET"])
@require_auth
def get_schedule():
    """Returns the current schedule settings, for the Schedule settings panel to populate itself."""
    cfg = load_schedule()
    if _scheduler:
        job = _scheduler.get_job("scrape_pipeline")
        cfg["next_scheduled_run"] = job.next_run_time.isoformat() if job and job.next_run_time else None
    else:
        cfg["next_scheduled_run"] = None
    return jsonify(cfg)


@app.route("/api/scrape/schedule", methods=["POST"])
@require_auth
def update_schedule():
    """
    Body: {"enabled": bool, "type": "cron"|"interval",
           "day_of_week": "mon", "hour": 3, "minute": 0,   # cron mode
           "interval_hours": 24}                            # interval mode
    Saves to Mongo and reschedules the live job immediately — no restart needed.
    """
    data = request.get_json(force=True, silent=True) or {}
    allowed = {"enabled", "type", "day_of_week", "hour", "minute", "interval_hours"}
    updates = {k: v for k, v in data.items() if k in allowed}

    if "type" in updates and updates["type"] not in ("cron", "interval"):
        return jsonify({"error": "type must be 'cron' or 'interval'"}), 400
    if "hour" in updates and not (0 <= int(updates["hour"]) <= 23):
        return jsonify({"error": "hour must be 0-23"}), 400
    if "minute" in updates and not (0 <= int(updates["minute"]) <= 59):
        return jsonify({"error": "minute must be 0-59"}), 400

    try:
        cfg = save_schedule(updates)
        apply_schedule(cfg)
        job = _scheduler.get_job("scrape_pipeline") if _scheduler else None
        cfg["next_scheduled_run"] = job.next_run_time.isoformat() if job and job.next_run_time else None
        return jsonify(cfg)
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

        # HSN codes stats
        _mongo_client_stats = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)
        db_stats = _mongo_client_stats[DB_NAME]

        hsn_col   = db_stats["hsn_codes"]
        total_hsn = hsn_col.count_documents({})
        hsn_chapters_done = len(hsn_col.distinct("chapter"))

        # Chapter extraction status breakdown
        notif_col = db_stats[COLLECTION]
        total_tariff_chapters = notif_col.count_documents({"category": "Tariff Schedule"})
        complete_chapters = notif_col.count_documents({
            "category": "Tariff Schedule",
            "hsn_extraction_complete": True
        })
        incomplete_chapters = notif_col.count_documents({
            "category": "Tariff Schedule",
            "hsn_extraction_complete": False
        })

        chunks_col = db_stats["chunks"]
        total_chunks = chunks_col.count_documents({})

        _mongo_client_stats.close()

        return jsonify({
            "total_notifications": total,
            "by_category":         by_cat,
            "total_hsn_codes":     total_hsn,
            "hsn_chapters_indexed": hsn_chapters_done,
            "total_tariff_chapters": total_tariff_chapters,
            "chapters_complete":   complete_chapters,
            "chapters_incomplete": incomplete_chapters,
            "total_chunks":        total_chunks,
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
    print("     POST /api/auth/signup  — create account")
    print("     POST /api/auth/login   — log in, get a token")
    print("     GET  /api/auth/me      — restore session from a stored token")
    print("     GET  /api/chats        — list your saved chat threads")
    print("     POST /api/chats        — create a new chat thread")
    print("     GET/PUT/DELETE /api/chats/<id> — load, save, or delete a thread")
    print("     POST /api/chat         — chat with uploaded PDF")
    print("     POST /api/chat/mongo   — chat with vector-search RAG over MongoDB")
    print("     GET  /api/search       — search notifications")
    print("     GET  /api/notification/<id> — get single doc")
    print("     POST /api/pdf/upload   — manually upload a PDF (auth required)")
    print("     POST /api/scrape/run   — trigger scrape+index+embed, non-blocking (auth required)")
    print("     GET  /api/scrape/status — poll progress of the last/current scrape run")
    print("     GET  /api/scrape/schedule — view the weekly/interval schedule (auth required)")
    print("     POST /api/scrape/schedule — update the schedule, takes effect immediately (auth required)")
    print("     GET  /api/stats        — DB stats\n")

    # In debug mode Flask's reloader forks a second process; only the forked
    # child sets WERKZEUG_RUN_MAIN, so gate the scheduler on that to avoid
    # starting it twice (which would double up scrape runs).
    if not is_dev or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_scheduler()

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