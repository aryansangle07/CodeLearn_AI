# CodeLearn AI 🧠

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.60-7928CA.svg)](https://github.com/langchain-ai/langgraph)
[![FAISS](https://img.shields.io/badge/FAISS-CPU_1.8-green.svg)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Deterministic Codebase Comprehension, Static AST Parsing & Semantic Retrieval Engine**

CodeLearn AI is a production-grade developer platform designed for zero-code-execution GitHub repository ingestion, AST symbol parsing, dual dense/sparse indexing (FAISS + BM25 with Reciprocal Rank Fusion), deterministic LangGraph orchestration, and grounded code question-answering with strict anti-hallucination source citation verification.

---

## 🌐 Live Interactive Demo

Try the live application hosted on Streamlit Cloud:
👉 **[https://codelearnai-lbsyydqmtlxxmuhsjka8nm.streamlit.app](https://codelearnai-lbsyydqmtlxxmuhsjka8nm.streamlit.app)**

---

## 🌟 Key Architecture & Capabilities

- **Zero-Code Execution Sandbox:** Extracts repository structures and abstracts symbols via Python AST parsing and Tree-sitter regex heuristic parsers without ever importing or executing foreign code.
- **Boundary & Secret Redaction:** Rejects oversized files (>2MB), archives (>50MB), and automatically sanitizes AWS keys, JWT tokens, private keys, and prompt injections before chunks reach the vector store.
- **Repository Health & Runnability Diagnostics:** Heuristically assesses whether a project is `RUNNABLE` 🟢, `PARTIALLY_RUNNABLE` 🟡, or `ANALYSIS_ONLY` 🔴 based on READMEs, dependency manifests, and entry points.
- **Hybrid Dense-Sparse Retrieval:** Combines FAISS `IndexFlatIP` (`all-MiniLM-L6-v2`, 384d) with code-tokenized `BM25Okapi` via Reciprocal Rank Fusion ($k=60$) with exact symbol score boosting ($1.5\times$).
- **Deterministic 4-Node LangGraph Pipeline:** Multi-agent state machine managing intent classification, hybrid retrieval, grounded answer generation within `<code_context>` XML tags, and backticked citation verification.
- **Pre-Seeded Demo Repositories & Auto-Cleanup:** Pre-indexes 3 curated, developer-friendly repositories (`pallets/click`, `psf/requests`, `encode/starlette`) protected by `is_demo = True`. Automatically purges ephemeral user-added session repositories via a TTL background cleaner.
- **Interactive Streamlit UI & FastAPI REST Gateway:** Real-time ingestion progress tracking, runnability breakdowns, and conversational Q&A with backticked source inspection.

---

## 🏗️ System Architecture & LangGraph Pipeline

```text
[Public GitHub Repository]
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│              Ingestion & Security Sandbox                   │
│  - Tarball Streaming & Anti-Zip Slip Extraction             │
│  - Boundary Filters (<2MB file, <50MB repo, max 500 files)  │
│  - High-Entropy Secret Redaction (AWS, Keys, JWTs)          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Static AST Parsing & Health                 │
│  - Tree-sitter & Python AST symbol extraction (defs, classes)│
│  - Semantic Chunking by function/class boundaries           │
│  - Health Diagnostics (RUNNABLE / PARTIAL / ANALYSIS_ONLY)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Hybrid Dual-Indexing Engine (RRF)              │
│  - Dense Index: FAISS IndexFlatIP (all-MiniLM-L6-v2, 384d)  │
│  - Sparse Index: BM25Okapi (code identifier tokenization)   │
│  - Reciprocal Rank Fusion (k=60, 1.5x symbol boost)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Deterministic 4-Node LangGraph Q&A Engine           │
│  [Intent Analysis] ➔ [Hybrid Retrieval] ➔ [Grounded LLM]    │
│                           ➔ [Citation Grounding Validator]  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### Option 1: 1-Click Docker Compose (Recommended for Local Full-Stack)

Start the PostgreSQL database, FastAPI REST backend, and Streamlit frontend in one command:

```bash
# 1. Clone repository
git clone https://github.com/aryansangle07/CodeLearn_AI.git
cd CodeLearn_AI

# 2. Configure Environment
cp .env.example .env
# Add your GROQ_API_KEY and GEMINI_API_KEY in .env

# 3. Spin up all services
docker compose up --build
```

- **Frontend Dashboard:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

---

### Option 2: Local Python Setup (Without Docker)

```bash
# 1. Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env to add your GROQ_API_KEY / GEMINI_API_KEY

# 4. Start the application
streamlit run app.py
```
*(Streamlit automatically spawns the FastAPI backend in a background daemon on port 8000).*

---

## 📦 Pre-Seeded Demo Repositories

CodeLearn AI includes built-in demo repositories pre-configured for evaluation:
1. **`encode/starlette`** — ASGI framework showcasing async AST routing and middleware structures.
2. **`pallets/click`** — Composable CLI toolkit showcasing decorator command trees and docstrings.
3. **`psf/requests`** — Python HTTP library showcasing modular architectures and sessions.

**How to Seed Repositories Locally:**
```bash
python seed_repos.py
```

---

## 🧪 Automated Testing Suite (48 / 48 Passed)

CodeLearn AI includes comprehensive automated regression tests covering all API endpoints, database operations, parsers, retrievers, LangGraph nodes, and security guardrails:

```bash
# Run the complete test suite
pytest -v --tb=short
```

**Test Coverage Summary:**
- `tests/test_api.py` (8/8 PASS) — REST endpoints, CORS, errors, status polling
- `tests/test_bm25_retriever.py` (3/3 PASS) — Tokenization, keyword search, index persistence
- `tests/test_citations.py` (3/3 PASS) — Citation extraction and anti-hallucination verification
- `tests/test_config.py` (3/3 PASS) — Settings resolution and database URL assembly
- `tests/test_db.py` (9/9 PASS) — SQLite & PostgreSQL CRUD, transaction rollbacks
- `tests/test_graph.py` (2/2 PASS) — LangGraph intent analysis and pipeline execution
- `tests/test_hybrid_retriever.py` (2/2 PASS) — Dual-index RRF ranking and symbol boost
- `tests/test_ingestion.py` (8/8 PASS) — Tarball security, regex redaction, prompt injection
- `tests/test_parser.py` (7/7 PASS) — Tree-sitter AST, chunking, health diagnostics
- `tests/test_vector_store.py` (3/3 PASS) — FAISS embeddings, dimensionality, vector CRUD

---

## 📂 Repository Layout

```text
CodeLearn_AI/
├── app.py                      # Streamlit Frontend Web Dashboard
├── main.py                     # FastAPI REST API Gateway & Lifespan
├── config.py                   # Pydantic BaseSettings Configuration
├── schemas.py                  # Pydantic Schemas & DTO Models
├── db.py                       # SQLAlchemy Models & Engine with SQLite Fallback
├── worker.py                   # Asynchronous Repo Ingestion & Chunking Worker
├── github_service.py           # GitHub Archive Streaming & Commit Resolution
├── file_filter.py              # File Type Whitelisting & Boundary Defense
├── parser.py                   # Tree-sitter & AST Code Symbol Parser
├── chunker.py                  # Semantic Code Chunking Engine
├── embedder.py                 # Dense Vector Embeddings (all-MiniLM-L6-v2)
├── vector_store.py             # FAISS Dense Index Storage
├── bm25_retriever.py           # Code-Tokenized BM25 Sparse Search
├── retriever.py                # Hybrid Dense-Sparse RRF Retriever
├── graph_state.py              # LangGraph Agent State TypedDict
├── intent_analyzer.py          # Deterministic Query Intent Classifier
├── citation_validator.py       # Grounding & Anti-Hallucination Validator
├── health_analyzer.py          # Codebase Health & Runnability Scoring
├── llm_service.py              # Gemini / Groq LLM Synthesis Service
├── graph.py                    # 4-Node LangGraph State Machine Assembly
├── security.py                 # Sandbox Security & Regex Secret Redaction
├── seed_service.py             # Demo Repository Pre-Seeding Service
├── seed_repos.py               # CLI Seeder Utility
├── tests/                      # 48 Automated Unit & Integration Tests
├── .github/workflows/ci.yml    # Automated GitHub Actions CI Pipeline
├── .python-version             # Python 3.11 Pinning for Cloud Deployment
├── packages.txt                # Linux C++ Dependencies (libgomp1)
├── requirements.txt            # Pinned Production Python Dependencies
├── docker-compose.yml          # Multi-Container Compose Configuration
├── Dockerfile.backend          # Multi-Stage Backend Dockerfile
├── Dockerfile.frontend         # Multi-Stage Frontend Dockerfile
├── .env.example                # Safe Configuration Template
├── .gitignore                  # Exclusion Rules (Keeps Secrets & Docs Local)
└── README.md                   # Project Documentation
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
