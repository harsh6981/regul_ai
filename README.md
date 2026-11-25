# RegulAI — Indian Trade & Customs Regulatory Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Gemini_2.5_Flash-free_tier-4285F4?logo=google&logoColor=white" alt="Gemini 2.5 Flash">
  <img src="https://img.shields.io/badge/MongoDB_Atlas-Vector_Search-47A248?logo=mongodb&logoColor=white" alt="MongoDB Atlas">
  <img src="https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white" alt="Flask">
</p>

> **AI-powered chatbot for Indian customs duties, HSN codes, trade notifications, and regulatory compliance** — built with Retrieval-Augmented Generation (RAG) over real CBIC data.

---

## ✨ What It Does

RegulAI scrapes, indexes, and answers questions about India's customs and trade regulations — the kind of questions that normally require a compliance officer to dig through hundreds of PDF notifications manually:

- 🔍 **"What is the duty rate for HSN code 3907 29 10?"** — instant structured lookup
- 📋 **"Compare chapter 29 and chapter 39 tariff schedules"** — side-by-side tables
- 📄 **"Summarize notification 50/2017-Customs"** — cited, grounded answers
- 🏷️ **"What's the HSN code for ferro-nickel?"** — reverse lookup by product name
- 💬 **Conversational follow-ups** — "what about the preferential rate?" works naturally

## 🏗️ Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│  CBIC Website │────▶│  project.py   │────▶│ CBIC_ALL_PDFS/   │
│  (cbic.gov.in)│     │  (Scraper)    │     │ (100+ PDFs)      │
└──────────────┘     └───────────────┘     └────────┬─────────┘
                                                     │
                     ┌───────────────┐               │
                     │mongo_pipeline │◀──────────────┘
                     │  (PDF → Mongo)│
                     └───────┬───────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
              ▼              ▼                  ▼
   ┌─────────────────┐ ┌──────────┐  ┌──────────────────┐
   │ embed_pipeline   │ │  hsn_    │  │ rule_based_      │
   │ (Sentence-BERT   │ │ extract_ │  │ extract.py       │
   │  → Atlas Vector)  │ │ pipeline │  │ (zero-LLM parse) │
   └────────┬─────────┘ └────┬─────┘  └────────┬─────────┘
            │                │                  │
            ▼                ▼                  ▼
   ┌─────────────────────────────────────────────────────┐
   │              MongoDB Atlas                          │
   │  ┌──────────────┐ ┌────────┐ ┌───────────────┐     │
   │  │ notifications │ │ chunks │ │  hsn_codes    │     │
   │  │ (full docs)   │ │(vectors)│ │ (structured)  │     │
   │  └──────────────┘ └────────┘ └───────────────┘     │
   └─────────────────────┬───────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │       app.py           │
            │  (Flask + RAG Chat)    │
            │  Gemini 2.5 Flash LLM  │
            └────────────┬───────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   Web UI (index.html)  │
            │   Dark-themed SPA      │
            └────────────────────────┘
```

## 📂 Project Structure

```
regul_ai/
├── chat_bot/
│   ├── app.py                    # Flask web server + API routes
│   ├── rag_search.py             # RAG engine — vector search, HSN/notification lookup, Gemini calls
│   ├── project.py                # CBIC website scraper — downloads PDFs + metadata
│   ├── mongo_pipeline.py         # PDF text extraction → MongoDB Atlas
│   ├── embed_pipeline.py         # Sentence-BERT embeddings → Atlas Vector Search index
│   ├── hsn_extract_pipeline.py   # Structured HSN code extraction (hybrid: rule-based + LLM)
│   ├── rule_based_extract.py     # Zero-LLM tariff table parser (state machine over PDF text)
│   ├── requirements.txt          # Python dependencies
│   ├── .env                      # API keys & MongoDB URI (not committed)
│   ├── cbic_master.json          # Scraped notification metadata
│   ├── CBIC_ALL_PDFS/            # Downloaded PDF files (not committed)
│   │   ├── chap-1.pdf ... chap-98.pdf    # Tariff Schedule chapters
│   │   └── *.pdf                          # Notification PDFs
│   └── static/
│       └── index.html            # Single-page web UI
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **MongoDB Atlas** account (free tier works) with [Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/) enabled
- **Google Gemini API Key** — free at [aistudio.google.com](https://aistudio.google.com/apikey) (no credit card required)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/regul_ai.git
cd regul_ai/chat_bot

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the `chat_bot/` directory:

```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?appName=regulai
GEMINI_API_KEY=AIzaSy...your-key-here
```

### 3. Run the Data Pipeline

```bash
# Step 1: Scrape CBIC notifications & download PDFs
python project.py

# Step 2: Extract PDF text → MongoDB
python mongo_pipeline.py

# Step 3: Build vector embeddings for RAG
python embed_pipeline.py

# Step 4: Extract structured HSN code data from tariff chapters
python hsn_extract_pipeline.py
```

> [!NOTE]
> Step 4 uses a hybrid approach — a rule-based parser handles most rows for free, with a single batched Gemini call per chapter to fix only the rows the parser couldn't confidently resolve. You can also run `--mode rule` for zero LLM calls, or `--mode llm` for full per-page extraction.

### 4. Set Up Atlas Vector Search Index

In your MongoDB Atlas dashboard, create a vector search index on the `chunks` collection:

```json
{
  "type": "vectorSearch",
  "fields": [
    {
      "path": "embedding",
      "numDimensions": 768,
      "type": "vector",
      "similarity": "cosine"
    }
  ]
}
```

Name the index `vector_index`.

### 5. Launch the App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Chat with an uploaded PDF (text passed from frontend) |
| `POST` | `/api/chat/mongo` | **RAG chat** — vector search + direct lookups over MongoDB |
| `GET`  | `/api/search?q=polymer&category=Import+Policy&limit=10` | Search notifications by keyword |
| `GET`  | `/api/notification/<id>` | Get a single notification (full text) |
| `GET`  | `/api/stats` | Database stats & health check |

### RAG Chat Request Example

```bash
curl -X POST http://localhost:5000/api/chat/mongo \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is HSN code 3907 29 10?"}
    ]
  }'
```

## 🧠 How the RAG Works

RegulAI uses a **multi-strategy retrieval** approach — it doesn't just do vector search:

1. **Direct notification lookup** — if the question mentions a notification number (e.g. "50/2017"), it fetches it directly from MongoDB, bypassing vector search entirely.

2. **Direct chapter lookup** — if the question names a tariff chapter ("chapter 39"), the chapter document is fetched by exact match.

3. **Structured HSN code lookup** — if the question contains an HSN code (4-8 digits), it queries the `hsn_codes` collection for structured data (description, unit, duty rates). Returns a **zero-LLM templated answer** for single codes, or a markdown table for multi-code comparisons.

4. **HSN name search** — "what's the HSN code for ferro-nickel?" does a regex search against descriptions, returning matching codes ranked by relevance.

5. **Atlas Vector Search** — for open-ended questions, embeds the query with the same Sentence-BERT model used for indexing (`BAAI/bge-base-en-v1.5`) and runs `$vectorSearch` against the `chunks` collection.

6. **Conversational follow-up resolution** — "what is the description?" after asking about code 2923 reuses the most recent entity from conversation history, so follow-ups don't fall through to vector search.

All retrieved context is passed to **Gemini 2.5 Flash** with a system prompt specialized for Indian trade law — it cites notification numbers, formats answers for compliance teams, and flags prohibitions/exemptions.

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MONGO_URI` | — | MongoDB Atlas connection string |
| `GEMINI_API_KEY` | — | Google AI Studio API key |
| `FLASK_ENV` | `production` | Set to `development` for debug mode |
| `PORT` | `5000` | Server port |
| `RATE_LIMIT_DEFAULT` | `60 per hour` | Global rate limit |
| `RATE_LIMIT_CHAT` | `20 per hour` | Chat endpoint rate limit |
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) |

## 🛡️ Security Notes

- **Rate limiting** is enabled by default (Flask-Limiter) to prevent API abuse
- **CORS** is wide-open by default for local dev; set `ALLOWED_ORIGINS` for production
- **Flask debug mode** is off by default; only enable via `FLASK_ENV=development`
- Production mode uses **Waitress** WSGI server instead of Flask's built-in dev server
- API keys are loaded from `.env` and never hardcoded

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Flask + Waitress (production WSGI) |
| **LLM** | Google Gemini 2.5 Flash (free tier, native `google-genai` SDK) |
| **Database** | MongoDB Atlas (documents + vector search) |
| **Embeddings** | Sentence-BERT (`BAAI/bge-base-en-v1.5`, 768-dim) |
| **PDF Parsing** | PyMuPDF (fitz) |
| **HSN Extraction** | Hybrid: rule-based state machine + Gemini structured output |
| **Frontend** | Vanilla HTML/CSS/JS single-page app |
| **Data Source** | [CBIC (cbic.gov.in)](https://www.cbic.gov.in) — Central Board of Indirect Taxes & Customs |

## 📊 Data Coverage

- **98 Tariff Schedule chapters** (HSN classification)
- **100+ customs/trade notifications** (duties, exemptions, anti-dumping, safeguards, circulars)
- **Categories**: Customs Duty, Import/Export Policy, Anti-Dumping, Safeguard, GST, Exemptions, Trade Notices, and more

## 🗺️ Roadmap

- [ ] Duty rate lookup for a specific date (historical rates)
- [ ] Duty rate trend analysis over time
- [ ] Multiple HSN code batch lookup improvements
- [ ] User authentication & saved queries
- [ ] Deployment to cloud (Render / Railway / GCP)

## 📄 License

This project is for educational and research purposes. CBIC data is publicly available government information.

---

<p align="center">
  Built with ❤️ for Indian trade compliance
</p>
