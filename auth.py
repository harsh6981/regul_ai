"""
auth.py — RegulAI authentication
=================================
Email + password auth backed by a MongoDB `users` collection, with
stateless JWT sessions. A JWT (rather than a server-side session store)
means app.py doesn't need Flask-Session/Redis — the token itself proves
who's asking. For the frontend, this also means "restoring a session
after refresh" is just: read the token out of localStorage on page
load, send it to GET /api/auth/me, and if that comes back 200 you're
still logged in as that email — no re-login needed.

Add to .env:
    JWT_SECRET=<any long random string>

Usage from app.py:
    from auth import create_user, verify_user, issue_token, require_auth
"""
from __future__ import annotations

import os
import re
import jwt
import datetime
from functools import wraps
from flask import request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

JWT_SECRET      = os.getenv("JWT_SECRET", "dev-only-change-me")
JWT_ALGO        = "HS256"
JWT_EXPIRY_DAYS = int(os.getenv("JWT_EXPIRY_DAYS", "7"))

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ─────────────────────────────────────────────────────────────
# Mongo
# ─────────────────────────────────────────────────────────────
_client    = None
_users_col = None


def get_users_col():
    """Lazy singleton, mirrors the pattern already used in app.py/embed_pipeline.py."""
    global _client, _users_col
    if _users_col is None:
        mongo_uri = os.getenv("MONGO_URI")
        _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5_000)
        db = _client["regulai"]
        _users_col = db["users"]
        _users_col.create_index("email", unique=True)
    return _users_col


# ─────────────────────────────────────────────────────────────
# Signup / login
# ─────────────────────────────────────────────────────────────
def create_user(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise ValueError("Please enter a valid email address")
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters")

    col = get_users_col()
    doc = {
        "email":         email,
        "password_hash": generate_password_hash(password),
        "created_at":    datetime.datetime.utcnow().isoformat(),
    }
    try:
        col.insert_one(doc)
    except DuplicateKeyError:
        raise ValueError("An account with this email already exists")
    return {"email": email}


def verify_user(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    col = get_users_col()
    user = col.find_one({"email": email})
    if not user or not check_password_hash(user["password_hash"], password):
        # Deliberately the same error for "no such user" and "wrong password"
        # so login can't be used to enumerate registered emails.
        raise ValueError("Invalid email or password")
    return {"email": email}


# ─────────────────────────────────────────────────────────────
# JWT
# ─────────────────────────────────────────────────────────────
def issue_token(email: str) -> str:
    payload = {
        "email": email,
        "iat":   datetime.datetime.utcnow(),
        "exp":   datetime.datetime.utcnow() + datetime.timedelta(days=JWT_EXPIRY_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])


def require_auth(fn):
    """
    Route decorator — rejects the request unless a valid
    'Authorization: Bearer <token>' header is present. On success,
    stashes the caller's email on request.user_email for the view to use.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth_header.split(" ", 1)[1].strip()
        try:
            payload = decode_token(token)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Session expired, please log in again"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid session token"}), 401
        request.user_email = payload["email"]
        return fn(*args, **kwargs)
    return wrapper
