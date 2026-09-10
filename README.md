# CodeLearn AI 🧠

**Deterministic Codebase Comprehension & Semantic Retrieval Engine**

CodeLearn AI is a production-grade developer tool designed for zero-code-execution GitHub repository ingestion, AST symbol parsing, dual dense/sparse indexing (FAISS + BM25 with Reciprocal Rank Fusion), deterministic LangGraph orchestration, and grounded code question-answering with strict anti-hallucination source citation verification.

---

## 🌟 Key Architecture & Capabilities

- **Zero-Code Execution Sandbox:** Extracts repository structures, abstracts symbols via Python AST parsing and regex heuristic parsers without ever executing foreign code.
- **Boundary & Secret Redaction:** Rejects oversized files (>500KB), archives (>50MB), and sanitizes AWS keys, JWT tokens, private keys, and prompt injections.
- **Repository Health & Runnability Diagnostics:** Heuristically assesses whether a project is `RUNNABLE` 🟢, `PARTIALLY_RUNNABLE` 🟡, or `ANALYSIS_ONLY` 🔴 based on READMEs, dependency manifests, and entry points.
- **Hybrid Dense-Sparse Retrieval:** Combines FAISS `IndexFlatIP` (`all-MiniLM-L6-v2`, 384d) with code-tokenized `BM25Okapi` via Reciprocal Rank Fusion ($k=60$) with exact symbol score boosting ($1.5\times$).
- **Deterministic 4-Node LangGraph Pipeline:** Intent analysis, hybrid retrieval, grounded answer generation within `<code_context>` XML tags, and backticked citation verification.
- **Interactive Streamlit UI & FastAPI REST Gateway:** Real-time ingestion progress tracking, runnability breakdowns, and conversational Q&A with source inspection.

---

## 🚀 Quick Start with Docker (Recommended)

Start the PostgreSQL database, FastAPI REST backend, and Streamlit frontend in one command:

```bash
# 1. Clone repository
git clone https://github.com/<YOUR_GITHUB_USERNAME>/CodeLearn_AI.git
cd CodeLearn_AI

# 2. Configure Environment
cp .env.example .env

# 3. Spin up all services
docker compose up --build
```

- **Frontend Dashboard:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

### 📦 Optional Demo Repository Auto-Seeding

For quick evaluation and demonstrations without manually ingesting a repository, CodeLearn AI provides an **opt-in auto-seeder**. It pre-indexes 3 curated, small, and clean developer repositories on startup:
1. `encode/starlette` (ASGI framework — showcases async AST, class hierarchies, and middleware routing).
2. `pallets/click` (CLI toolkit — showcases decorator-based command trees and rich docstring chunks).
3. `psf/requests` (HTTP library — showcases modular functions and session classes).

**How to Enable/Disable:**
- **In Docker Compose:** Set `SEED_DEMO_REPOS=true` in `.env` before running `docker compose up --build` (default: `false`).
- **In Local CLI:** Run `python seed_repos.py` directly to pre-seed your local database.
- *Fault Tolerance:* If any repository encounters a network or rate limit issue, the seeder logs a warning and continues cleanly without blocking server startup.

---

## 💻 Local Development Setup (Without Docker)

### 1. Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create `.env` from `.env.example`:
```env
DATABASE_URL=sqlite:///./codelearn.db
INDEX_STORAGE_DIR=./indexes
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.8-27b
```

### 3. Run FastAPI Backend
```bash
uvicorn main:app --reload --port 8000
```

### 4. Run Streamlit Frontend
```bash
streamlit run app.py
```

---

## 🧪 Testing & Verification

Execute the complete 45-test automated regression suite:
```bash
pytest -v --tb=short
```

Execute individual Phase User Acceptance Testing scripts:
```bash
python run_uat_phase_1.py
python run_uat_phase_2.py
python run_uat_phase_3.py
python run_uat_phase_4.py
python run_uat_phase_5.py
python run_uat_phase_6.py
```

---

## 📚 Technical Documentation

Detailed specifications and architectural contracts are available in [`docs/`](./docs/):
- [`docs/system_architecture.md`](./docs/system_architecture.md) — Master System Architecture Contract
- [`docs/PROJECT_STATUS.md`](./docs/PROJECT_STATUS.md) — Phase Implementation Status
- [`docs/phase_1_foundation.md`](./docs/phase_1_foundation.md) — Phase 1: Database & Foundation
- [`docs/phase_2_ingestion_and_parsing.md`](./docs/phase_2_ingestion_and_parsing.md) — Phase 2: Ingestion & AST Parsing
- [`docs/phase_3_vector_and_retrieval.md`](./docs/phase_3_vector_and_retrieval.md) — Phase 3: Hybrid FAISS & BM25 Indexing
- [`docs/phase_4_langgraph_and_grounding.md`](./docs/phase_4_langgraph_and_grounding.md) — Phase 4: Deterministic LangGraph
- [`docs/phase_5_api_and_ui.md`](./docs/phase_5_api_and_ui.md) — Phase 5: FastAPI & Streamlit UI
- [`docs/phase_6_devops_and_ci.md`](./docs/phase_6_devops_and_ci.md) — Phase 6: DevOps, Docker & CI/CD
