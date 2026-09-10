import os
import time
from typing import Any, Dict, List, Optional
import httpx
import streamlit as st

# Ensure SQLite is default for zero-dependency standalone Streamlit Cloud deployments
if "DATABASE_URL" not in os.environ or not os.environ["DATABASE_URL"]:
    os.environ["DATABASE_URL"] = "sqlite:///./data/codelearn_ai.db"

# Page Configuration
st.set_page_config(
    page_title="CodeLearn AI — Codebase Comprehension",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and clean typography
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .badge-runnable {
        background-color: #065f46;
        color: #34d399;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-partial {
        background-color: #78350f;
        color: #fbbf24;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-analysis {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .citation-pill {
        background-color: #0f172a;
        border: 1px solid #38bdf8;
        color: #38bdf8;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        margin-right: 0.4rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Backend Client Initialization (In-Process ASGI for Streamlit Cloud & Standalone Deployment)
@st.cache_resource
def get_backend_client():
    """Initializes and caches the in-process FastAPI ASGI client with lifecycle management."""
    try:
        from db import get_engine, init_db
        engine = get_engine()
        init_db(engine)
    except Exception:
        pass

    from main import app as fastapi_app
    from fastapi.testclient import TestClient
    client = TestClient(fastapi_app)
    client.__enter__()
    return client


def fetch_api(endpoint: str, method: str = "GET", json_data: Optional[Dict[str, Any]] = None):
    """Communicates with FastAPI backend via in-process ASGI client or HTTP fallback."""
    try:
        client = get_backend_client()
        if method == "POST":
            resp = client.post(endpoint, json=json_data)
        elif method == "DELETE":
            resp = client.delete(endpoint)
        elif method == "PUT":
            resp = client.put(endpoint, json=json_data)
        else:
            resp = client.get(endpoint)

        if resp.status_code in (200, 201, 202):
            return resp.json()
        elif resp.status_code == 403:
            return {"error": resp.json().get("detail", "Operation forbidden.")}
        elif resp.status_code == 404:
            return {"error": resp.json().get("detail", "Resource not found.")}
        else:
            try:
                err_detail = resp.json().get("detail", resp.text)
            except Exception:
                err_detail = resp.text
            return {"error": f"HTTP {resp.status_code}: {err_detail}"}
    except Exception as exc:
        return {"error": f"Backend execution error: {str(exc)}"}


# Session State Initialization
if "selected_repo_id" not in st.session_state:
    st.session_state.selected_repo_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "confirm_del_repo_id" not in st.session_state:
    st.session_state.confirm_del_repo_id = None
if "input_url_val" not in st.session_state:
    st.session_state.input_url_val = ""
if "input_branch_val" not in st.session_state:
    st.session_state.input_branch_val = "main"


def trigger_ingestion(url: str, branch: str = "main"):
    """Helper function to trigger live repository ingestion and progress tracking."""
    with st.spinner("Submitting repository to ingestion engine..."):
        res = fetch_api(
            "/api/repositories",
            method="POST",
            json_data={"github_url": url.strip(), "branch": branch.strip() or "main"},
        )
        if "error" in res:
            st.error(res["error"])
        else:
            repo_id = res["repository_id"]
            st.session_state.selected_repo_id = repo_id
            st.session_state.chat_history = []
            st.success(res["message"])

            # Poll progress if not already cached
            if not res.get("is_cached", False):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for _ in range(60):
                    time.sleep(1.0)
                    status_res = fetch_api(f"/api/repositories/{repo_id}/status")
                    if "error" in status_res:
                        status_text.error(status_res["error"])
                        break

                    pct = status_res.get("progress_percentage", 0)
                    step = status_res.get("current_step", "")
                    st_val = status_res.get("status", "")

                    progress_bar.progress(pct)
                    status_text.info(f"**Status:** {st_val} — {step}")

                    if st_val == "INDEXED":
                        st.success("🎉 Ingestion and Dual Indexing Complete!")
                        time.sleep(0.5)
                        st.rerun()
                        break
                    elif st_val == "FAILED":
                        st.error(f"❌ Ingestion Failed: {status_res.get('error_message')}")
                        break


# --- SIDEBAR: Ingestion & Repository Selection ---
with st.sidebar:
    st.markdown("### 📥 Ingest Repository")
    st.caption("Enter any public GitHub repository URL/link to parse, chunk, and index:")
    
    repo_url_input = st.text_input(
        "Enter GitHub repository URL / link:",
        value=st.session_state.input_url_val,
        placeholder="https://github.com/owner/repository",
        help="Paste a public GitHub repository link (e.g. https://github.com/encode/starlette) to analyze.",
    )
    branch_input = st.text_input(
        "Branch / Tag:",
        value=st.session_state.input_branch_val,
        help="Git branch or tag (defaults to main).",
    )

    if st.button("🚀 Ingest & Index Codebase", use_container_width=True, type="primary"):
        if not repo_url_input.strip():
            st.warning("⚠️ Please enter a GitHub repository URL/link above.")
        else:
            trigger_ingestion(repo_url_input, branch_input)

    # --- 1-CLICK QUICK TRY PRESET ---
    st.markdown("---")
    st.markdown("##### ⚡ Quick Start Preset")
    st.caption("Don't have a repository link handy? Try our verified lightweight preset:")
    if st.button("⚡ 1-Click Try: encode/starlette", use_container_width=True):
        st.session_state.input_url_val = "https://github.com/encode/starlette"
        st.session_state.input_branch_val = "master"
        trigger_ingestion("https://github.com/encode/starlette", "master")

    # Fetch current session repositories
    repos_list = fetch_api("/api/repositories")
    indexed_repos_app = [r for r in repos_list if r.get("status") == "INDEXED"] if isinstance(repos_list, list) else []
    pending_repos_app = [r for r in repos_list if r.get("status") != "INDEXED"] if isinstance(repos_list, list) else []

    # Default selection to latest ingested repository if current is not set
    if not st.session_state.selected_repo_id and indexed_repos_app:
        st.session_state.selected_repo_id = indexed_repos_app[0]["id"]

    # --- SECTION: INGESTED REPOSITORIES DROPDOWN ---
    st.markdown("---")
    st.markdown("### 📁 Ingested Repositories")
    if indexed_repos_app:
        repo_options = {f"📦 {r['owner']}/{r['name']} ({r.get('total_files', 0)} files)": r["id"] for r in indexed_repos_app}
        repo_labels = list(repo_options.keys())

        active_idx = 0
        for idx, r_id in enumerate(repo_options.values()):
            if r_id == st.session_state.selected_repo_id:
                active_idx = idx
                break

        def on_repo_change():
            selected_label = st.session_state.ingested_repo_selector
            st.session_state.selected_repo_id = repo_options[selected_label]
            st.session_state.chat_history = []

        selected_repo_label = st.selectbox(
            "Select Active Repository:",
            options=repo_labels,
            index=active_idx,
            key="ingested_repo_selector",
            on_change=on_repo_change,
            help="Choose which ingested repository to inspect and query.",
        )

        chosen_id = repo_options[selected_repo_label]
        if st.session_state.selected_repo_id != chosen_id:
            st.session_state.selected_repo_id = chosen_id
            st.session_state.chat_history = []
            st.rerun()

        active_repo = next((r for r in indexed_repos_app if r["id"] == st.session_state.selected_repo_id), indexed_repos_app[0])
        with st.container(border=True):
            if st.session_state.confirm_del_repo_id != active_repo["id"]:
                st.markdown(f"**Active:** `{active_repo['owner']}/{active_repo['name']}`")
                st.caption(f"📁 Files: {active_repo.get('total_files', 0)} | 🔖 Commit: `{active_repo.get('commit_sha', '')[:7]}` | ⚡ Health: `{active_repo.get('runnability_score', '')}`")
                if st.button("🗑️ Delete Repository", key="btn_del_repo", use_container_width=True):
                    st.session_state.confirm_del_repo_id = active_repo["id"]
                    st.rerun()
            else:
                st.error(f"⚠️ **Confirm Deletion**\n\nPermanently delete **`{active_repo['owner']}/{active_repo['name']}`**?")
                c_yes, c_no = st.columns([1, 1])
                with c_yes:
                    if st.button("🗑️ Yes, Delete", key="btn_yes_del_repo", type="primary", use_container_width=True):
                        fetch_api(f"/api/repositories/{active_repo['id']}", method="DELETE")
                        st.session_state.selected_repo_id = None
                        st.session_state.confirm_del_repo_id = None
                        st.session_state.chat_history = []
                        st.toast("Repository deleted!")
                        st.rerun()
                with c_no:
                    if st.button("Cancel", key="btn_cancel_del_repo", use_container_width=True):
                        st.session_state.confirm_del_repo_id = None
                        st.rerun()

        if len(indexed_repos_app) > 1:
            if st.button("🧹 Clean All Repositories", key="btn_clean_all", use_container_width=True):
                fetch_api("/api/cleanup-session", method="POST")
                st.session_state.selected_repo_id = None
                st.session_state.confirm_del_repo_id = None
                st.session_state.chat_history = []
                st.toast("All repositories cleaned!")
                st.rerun()
    else:
        st.info("ℹ️ No repositories ingested yet. Click **⚡ 1-Click Try: encode/starlette** above or enter a GitHub link to start!")

    # In-flight ingestion tracking
    if pending_repos_app:
        st.markdown("---")
        st.caption("⏳ **In-Flight Background Ingestion:**")
        for pr in pending_repos_app:
            st.info(f"⚙️ **{pr['owner']}/{pr['name']}**: `{pr.get('status')}`")


# --- MAIN CONTENT AREA ---
st.markdown('<div class="main-header">CodeLearn AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Zero-Code-Execution Repository Comprehension & Grounded Semantic Retrieval</div>',
    unsafe_allow_html=True,
)

# Validate that selected_repo_id is among current indexed repos
indexed_ids = [r["id"] for r in indexed_repos_app]
if st.session_state.selected_repo_id not in indexed_ids:
    st.session_state.selected_repo_id = indexed_ids[0] if indexed_ids else None

if not st.session_state.selected_repo_id:
    # Beautiful Onboarding Hero View
    with st.container(border=True):
        st.markdown("### 👋 Welcome to CodeLearn AI")
        st.markdown(
            "CodeLearn AI is a zero-code-execution developer platform for static repository comprehension, AST symbol parsing, and grounded code retrieval."
        )
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("#### 🛡️ Safe AST Parsing")
            st.caption("Extracts functions, classes, and call structures using Python AST & Tree-sitter without ever executing foreign code.")
        with col_b:
            st.markdown("#### 🔍 Hybrid Retrieval")
            st.caption("Combines FAISS dense embeddings (all-MiniLM-L6-v2) and BM25 sparse keyword search via Reciprocal Rank Fusion.")
        with col_c:
            st.markdown("#### 💬 Grounded Q&A")
            st.caption("LangGraph agent state machine generates answers strictly grounded in indexed code with verified citations.")
        
        st.markdown("---")
        st.markdown("#### 🚀 Get Started Now:")
        btn_c1, btn_c2 = st.columns([1, 2])
        with btn_c1:
            if st.button("⚡ Launch Preset: encode/starlette", key="btn_hero_launch", type="primary", use_container_width=True):
                st.session_state.input_url_val = "https://github.com/encode/starlette"
                st.session_state.input_branch_val = "master"
                trigger_ingestion("https://github.com/encode/starlette", "master")
        with btn_c2:
            st.caption("👈 Or enter any public GitHub repository link in the sidebar to parse and query your own codebase.")
else:
    active_repo_id = st.session_state.selected_repo_id
    summary = fetch_api(f"/api/repositories/{active_repo_id}/summary")

    if "error" in summary:
        st.session_state.selected_repo_id = None
        st.session_state.chat_history = []
        st.warning("👈 Repository not found or was removed. Please select an active repository from the sidebar.")
        time.sleep(0.5)
        st.rerun()
    else:
        tab_overview, tab_qa = st.tabs(["📊 Repository Health & Overview", "💬 Grounded Q&A Chat"])

        # --- TAB 1: OVERVIEW & HEALTH ---
        with tab_overview:
            # Metric Columns
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Repository", f"{summary['owner']}/{summary['name']}")
            c2.metric("Total Processed Files", f"{summary['total_files']:,}")
            c3.metric("Total Lines of Code", f"{summary['total_lines']:,}")
            c4.metric("Extracted Size", f"{summary['total_size_bytes'] / 1024:.1f} KB")

            st.markdown("---")
            h_col1, h_col2 = st.columns([1, 1])

            with h_col1:
                st.markdown("#### 🛡️ Runnability Health Assessment")
                runnability = summary.get("runnability_score", "ANALYSIS_ONLY")
                if runnability == "RUNNABLE":
                    badge_html = '<span class="badge-runnable">🟢 FULLY RUNNABLE</span>'
                elif runnability == "PARTIALLY_RUNNABLE":
                    badge_html = '<span class="badge-partial">🟡 PARTIALLY RUNNABLE</span>'
                else:
                    badge_html = '<span class="badge-analysis">🔴 ANALYSIS ONLY</span>'

                st.markdown(badge_html, unsafe_allow_html=True)
                health = summary.get("health_details", {})
                st.write(f"**Assessment:** {health.get('summary_assessment', 'N/A')}")

                st.markdown("##### Diagnostics:")
                st.write(f"- **Documentation:** {'✅ README detected' if health.get('has_readme') else '❌ Missing README'}")
                st.write(f"- **Dependencies:** {'✅ Manifests: ' + ', '.join(health.get('dependency_files', [])) if health.get('has_dependency_manifest') else '❌ No manifest found'}")
                st.write(f"- **Entry Points:** {'✅ Entry files: ' + ', '.join(health.get('entry_point_files', [])) if health.get('has_entry_point') else '❌ No main entry point'}")
                st.write(f"- **Empty Stubs / TODOs:** {health.get('empty_implementations_count', 0)} stubs | {health.get('todo_comment_count', 0)} TODOs")

            with h_col2:
                st.markdown("#### 🌐 Language Distribution")
                lang_dist = summary.get("language_distribution", {})
                if lang_dist:
                    for lang, pct in lang_dist.items():
                        st.write(f"**{lang}:** {pct}%")
                        st.progress(int(pct))
                else:
                    st.info("No language distribution data.")

        # --- TAB 2: GROUNDED Q&A CHAT ---
        with tab_qa:
            st.markdown("#### 🤖 Grounded Codebase Assistant")
            st.markdown("Ask anything about functions, classes, architecture, or workflows. All answers are strictly grounded in indexed code.")

            # Display Chat History
            for item in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(item["query"])
                with st.chat_message("assistant"):
                    st.markdown(item["answer"])

                    # Render Citation Pills
                    if item.get("citations"):
                        pills = "".join([f'<span class="citation-pill">{c}</span>' for c in item["citations"]])
                        st.markdown(f"**Verified Sources:** {pills}", unsafe_allow_html=True)

                    # Retrieved Chunks Expander
                    if item.get("retrieved_chunks"):
                        with st.expander("🔍 View Retrieved Grounding Chunks"):
                            for ch in item["retrieved_chunks"]:
                                st.markdown(f"**File:** `{ch['file_path']}` (Lines {ch['start_line']}-{ch['end_line']}) | **Symbol:** `{ch['symbol_name']}`")
                                st.code(ch["content"], language="python")

            # Chat Input Box
            query_prompt = st.chat_input("Ask a question about this repository...")
            if query_prompt:
                # Add user query to chat
                with st.chat_message("user"):
                    st.write(query_prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing codebase & retrieving grounded evidence..."):
                        q_res = fetch_api(
                            "/api/query",
                            method="POST",
                            json_data={
                                "repository_id": active_repo_id,
                                "query": query_prompt,
                            },
                        )

                        if "error" in q_res:
                            st.error(q_res["error"])
                        else:
                            answer = q_res.get("answer", "No answer returned.")
                            citations = q_res.get("citations", [])
                            chunks = q_res.get("retrieved_chunks", [])
                            exec_time = q_res.get("execution_time_seconds", 0.0)

                            st.markdown(answer)

                            if citations:
                                pills = "".join([f'<span class="citation-pill">{c}</span>' for c in citations])
                                st.markdown(f"**Verified Sources:** {pills}", unsafe_allow_html=True)

                            st.caption(f"⏱️ Executed in {exec_time}s | Intent: `{q_res.get('intent_type')}` | Grounded: `{'✅ Yes' if q_res.get('is_grounded') else '⚠️ Partially Verified'}`")

                            if chunks:
                                with st.expander("🔍 View Retrieved Grounding Chunks"):
                                    for ch in chunks:
                                        st.markdown(f"**File:** `{ch['file_path']}` (Lines {ch['start_line']}-{ch['end_line']}) | **Symbol:** `{ch['symbol_name']}`")
                                        st.code(ch["content"], language="python")

                            # Save to session chat history
                            st.session_state.chat_history.append({
                                "query": query_prompt,
                                "answer": answer,
                                "citations": citations,
                                "retrieved_chunks": chunks,
                            })
