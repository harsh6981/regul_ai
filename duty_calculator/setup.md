# Duty Calculator — Setup

This folder ports ICEDutyAI's duty cascade calculator into RegulAI. No new
API keys needed — it reuses your existing `MONGO_URI` and `GEMINI_API_KEY`.

## 1. Install the one new dependency

```bash
pip install -r requirements.txt   # adds `cachetools` for the ICEGATE rate cache
```

## 2. One-time data migration

`dutyai_source.db` (bundled here) is ICEDutyAI's original SQLite reference
data: 1,192 HS codes (CUSTADA baseline rates) + 187 anti-dumping/
countervailing/safeguard measures. Load it into your `regulai` MongoDB as
two new collections (`duty_hs_codes`, `trade_defense`):

```bash
python duty_calculator/migrate_sqlite_to_mongo.py
```

Safe to re-run any time (upserts by primary key). Note: this also fixes a
data bug in the source file where 84 HS codes (mostly chapters 1–9) had
their leading zero stripped (e.g. `1010000` instead of `01010000`) —
the migration script left-pads these so lookups work correctly.

You can drop `dutyai_source.db` after migrating once it's confirmed in
Mongo — it isn't read by the app itself, only by the migration script.

## 3. Start the app as usual

```bash
python -u app.py
```

## New endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/duty/countries` | Supported country codes |
| POST | `/api/duty/classify` | AI product description → suggested CTH codes (Gemini) |
| GET | `/api/duty/duty-structure?cth=&country=` | Live ICEGATE rates (or fallback estimates) |
| GET | `/api/duty/fta-check?cth=&country=` | FTA eligibility |
| GET | `/api/duty/trade-defense/<hs_code>?country=` | ADD/CVD/Safeguard measures |
| GET | `/api/duty/hs-lookup/<hs_code>` | CUSTADA baseline rates for one HS code |
| GET | `/api/duty/hs-search?q=&limit=` | Search HS codes by description |
| POST | `/api/duty/calculate` | Full pipeline: rates → cascade → FTA → trade defense → AI summary |
| GET | `/api/duty/db-stats` | Row counts for the migrated reference data |

`/api/duty/calculate` body:
```json
{ "cth": "39076190", "country": "USA", "cif_value": 500000, "quantity": null }
```

`/calculate` and `/classify` are rate-limited (`RATE_LIMIT_DUTY_CALCULATE`,
`RATE_LIMIT_DUTY_CLASSIFY` env vars, default 20/hour each) since they call
Gemini and/or scrape ICEGATE live — same pattern as `/api/chat/mongo`. They
are **not** gated behind `@require_auth`; add that decorator in `app.py` if
you want the calculator restricted to logged-in RegulAI users only.

## What changed vs. the original ICEDutyAI backend

- **AI (classify/summary):** now calls Gemini via RegulAI's existing
  `call_gemini()` instead of Anthropic Claude Haiku — no new SDK or API key.
- **Scraper:** converted from async `httpx` (FastAPI) to sync `requests`
  (Flask). Same multi-step ICEGATE navigation and HTML parsing.
- **Data store:** SQLite → MongoDB (`duty_hs_codes`, `trade_defense`
  collections in your existing `regulai` database), so there's one
  datastore instead of two.
- **Auth:** ICEDutyAI's separate API-key/tier/quota system (for external
  API consumers) wasn't ported — RegulAI's existing JWT auth + Flask-Limiter
  cover the same ground for an app with its own logged-in users. The
  original `auth.py`/`monitoring.py` files are in the ICEDutyAI zip if you
  want that system later for third-party API access.
- **Not ported:** ICEDutyAI's React/Vite/shadcn frontend (different stack
  entirely). A new **"Duty Calculator" tab** was instead built directly into
  `static/index.html`/`style.css`, matching RegulAI's existing dark theme —
  see below.

## Frontend

A new "Chat / Duty Calculator" switch sits at the top of the sidebar. The
Duty Calculator view has:

- Product description + **AI Classify** button (suggests CTH codes via Gemini)
- 8-digit CTH field (fillable manually or by clicking a suggestion), country
  dropdown, CIF value field
- Results: AI plain-English summary, FTA eligibility banner, any applicable
  trade-defense (ADD/CVD/Safeguard) measures, and the full 14-step duty
  breakdown table

It's entirely separate from the chat UI — switching views just toggles which
panel is visible inside `#main`; nothing about the existing chat logic
changed.