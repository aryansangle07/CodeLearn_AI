import time
from typing import Any, Dict, List, Optional
import httpx
import streamlit as st

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
    from main import app as fastapi_app
    from fastapi.testclient import TestClient
    client = TestClient(fastapi_app)
    client.__enter__()
    return client


def fetch_api(endpoint: str, method: str = "GET", json_data: Optional[Dict[str, Any]] = None):
    """Communicates with FastAPI backend via in-process ASGI client or HTTP fallback."""
    # 1. Direct In-Process ASGI execution (fast, zero socket errors on Streamlit Cloud)
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


# --- SIDEBAR: Ingestion & Repository Selection ---
with st.sidebar:
    st.markdown("### 📥 Repository Ingestion")
    repo_url_input = st.text_input(
        "GitHub Repository URL",
        value="https://github.com/pallets/flask",
        placeholder="https://github.com/owner/repo",
    )
    branch_input = st.text_input("Branch / Tag", value="main")

    if st.button("🚀 Ingest & Index Codebase", use_container_width=True):
        if repo_url_input:
            with st.spinner("Submitting repository to ingestion engine..."):
                res = fetch_api(
                    "/api/repositories",
                    method="POST",
                    json_data={"github_url": repo_url_input, "branch": branch_input},
                )
                if "error" in res:
                    st.error(res["error"])
                else:
                    repo_id = res["repository_id"]
                    st.session_state.selected_repo_id = repo_id
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
                                break
                            elif st_val == "FAILED":
                                st.error(f"❌ Ingestion Failed: {status_res.get('error_message')}")
                                break

    st.markdown("---")
    st.markdown("### 📂 Ingested Repositories")
    repos_list = fetch_api("/api/repositories")

    if isinstance(repos_list, list) and repos_list:
        indexed_repos_app = [r for r in repos_list if r.get("status") == "INDEXED"]
        pending_repos_app = [r for r in repos_list if r.get("status") != "INDEXED"]

        if indexed_repos_app:
            repo_options = {}
            for r in indexed_repos_app:
                tag = "⭐ [Demo]" if r.get("is_demo") else "👤 [Session]"
                label = f"{tag} {r['owner']}/{r['name']} ({r.get('total_files', 0)} files)"
                repo_options[label] = r["id"]
            repo_keys = list(repo_options.keys())

            # Find active index
            active_idx = 0
            if st.session_state.selected_repo_id:
                for i, r_id in enumerate(repo_options.values()):
                    if r_id == st.session_state.selected_repo_id:
                        active_idx = i
                        break

            selected_label = st.selectbox(
                "Select Repository:",
                options=repo_keys,
                index=active_idx,
                help="Choose an ingested repository to inspect and query.",
            )
            st.session_state.selected_repo_id = repo_options[selected_label]
            active_repo = next(r for r in indexed_repos_app if r["id"] == st.session_state.selected_repo_id)

            # Details card with Protected Badge for Demo or Delete Button for Session Repos
            with st.container(border=True):
                is_demo_repo = active_repo.get("is_demo", False)
                if is_demo_repo:
                    st.markdown(f"**Active:** `{active_repo['owner']}/{active_repo['name']}` ⭐ *(Default Demo)*")
                    st.caption(f"📁 Files: {active_repo.get('total_files', 0)} | 🔖 Commit: `{active_repo.get('commit_sha', '')[:7]}` | 🛡️ **Protected (Q&A Enabled)**")
                    st.info("🛡️ This demo repository is pre-indexed and protected from deletion. You can freely query and inspect its architecture.")
                elif st.session_state.confirm_del_repo_id != active_repo["id"]:
                    st.markdown(f"**Active:** `{active_repo['owner']}/{active_repo['name']}` 👤 *(User Session)*")
                    st.caption(f"📁 Files: {active_repo.get('total_files', 0)} | 🔖 Commit: `{active_repo.get('commit_sha', '')[:7]}` | ⚡ Health: `{active_repo.get('runnability_score', '')}`")
                    if st.button("🗑️ Delete Session Repo", key="btn_del_active_repo", use_container_width=True):
                        st.session_state.confirm_del_repo_id = active_repo["id"]
                        st.rerun()
                else:
                    # Windows-style confirmation card inside the active container
                    st.error(
                        f"⚠️ **Confirm Permanent Deletion**\n\n"
                        f"Are you sure you want to permanently delete **`{active_repo['owner']}/{active_repo['name']}`**?\n\n"
                        f"*This will remove all vector indices, chunks, and metadata from storage.*"
                    )
                    c_yes, c_no = st.columns([1, 1])
                    with c_yes:
                        if st.button("🗑️ Yes, Delete", key="btn_yes_delete_repo", type="primary", use_container_width=True):
                            with st.spinner("Deleting repository from storage..."):
                                del_res = fetch_api(f"/api/repositories/{active_repo['id']}", method="DELETE")
                                if "error" in del_res:
                                    st.error(del_res["error"])
                                else:
                                    st.session_state.selected_repo_id = None
                                    st.session_state.confirm_del_repo_id = None
                                    st.session_state.chat_history = []
                                    st.toast(f"Successfully deleted {active_repo['owner']}/{active_repo['name']}")
                                    time.sleep(0.5)
                                    st.rerun()
                    with c_no:
                        if st.button("Cancel", key="btn_cancel_delete_repo", use_container_width=True):
                            st.session_state.confirm_del_repo_id = None
                            st.rerun()

            # Global Clean Custom Session Repos Button
            user_session_repos = [r for r in indexed_repos_app if not r.get("is_demo")]
            if user_session_repos:
                if st.button("🧹 Clean All User Session Repos", key="btn_clean_all_session", use_container_width=True):
                    with st.spinner("Purging temporary session repositories..."):
                        clean_res = fetch_api("/api/cleanup-session", method="POST")
                        if "error" in clean_res:
                            st.error(clean_res["error"])
                        else:
                            st.session_state.selected_repo_id = None
                            st.session_state.confirm_del_repo_id = None
                            st.session_state.chat_history = []
                            st.toast("Temporary user session repositories cleaned!")
                            time.sleep(0.5)
                            st.rerun()
        else:
            st.info("No repositories have completed indexing yet.")

        # Display in-flight auto-seeding or ingestion jobs
        if pending_repos_app:
            st.markdown("---")
            st.caption("⏳ **In-Flight Background Ingestion:**")
            for pr in pending_repos_app:
                st.info(f"⚙️ **{pr['owner']}/{pr['name']}**: `{pr.get('status')}`")
            if st.button("🔄 Refresh Repository List", key="btn_refresh_sidebar", use_container_width=True):
                st.rerun()
    else:
        st.info("⚙️ Initializing repository storage...")
        if st.button("🔄 Refresh", key="btn_init_refresh", use_container_width=True):
            st.rerun()


# --- MAIN CONTENT AREA ---
st.markdown('<div class="main-header">CodeLearn AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Zero-Code-Execution Repository Comprehension & Grounded Semantic Retrieval</div>',
    unsafe_allow_html=True,
)

# Validate that selected_repo_id is among current indexed repos
repos_list_main = fetch_api("/api/repositories")
indexed_ids = [r["id"] for r in repos_list_main if r.get("status") == "INDEXED"] if isinstance(repos_list_main, list) else []

if st.session_state.selected_repo_id not in indexed_ids:
    st.session_state.selected_repo_id = indexed_ids[0] if indexed_ids else None

if not st.session_state.selected_repo_id:
    st.warning("👈 Please enter a GitHub repository URL in the sidebar to begin ingestion or select an indexed repo.")
else:
    active_repo_id = st.session_state.selected_repo_id
    summary = fetch_api(f"/api/repositories/{active_repo_id}/summary")

    if "error" in summary:
        # Gracefully handle stale or deleted repository IDs
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
