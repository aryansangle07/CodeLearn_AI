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
    """Initializes and caches the in-process FastAPI ASGI client with lifecycle management and DB seeding."""
    try:
        from db import get_engine, init_db
        from seed_service import run_demo_seed_pipeline
        engine = get_engine()
        init_db(engine)
        run_demo_seed_pipeline(background=False, engine=engine)
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


# --- SIDEBAR: Ingestion & Repository Selection ---
with st.sidebar:
    st.markdown("### 📥 Ingest Custom Repository")
    st.caption("Enter any public GitHub repository URL/link below to parse, chunk, and index:")
    repo_url_input = st.text_input(
        "Enter GitHub repository URL / link:",
        value="",
        placeholder="https://github.com/owner/repository",
        help="Paste a public GitHub repository link (e.g. https://github.com/encode/starlette) to analyze.",
    )
    branch_input = st.text_input("Branch / Tag:", value="main", help="Git branch or tag (defaults to main).")

    if st.button("🚀 Ingest & Index Codebase", use_container_width=True, type="primary"):
        if not repo_url_input.strip():
            st.warning("⚠️ Please enter a GitHub repository URL/link in the query bar above.")
        else:
            with st.spinner("Submitting repository to ingestion engine..."):
                res = fetch_api(
                    "/api/repositories",
                    method="POST",
                    json_data={"github_url": repo_url_input.strip(), "branch": branch_input.strip() or "main"},
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

    repos_list = fetch_api("/api/repositories")

    # If backend returned an error or empty list, query direct DB / seed
    if not isinstance(repos_list, list) or len(repos_list) == 0:
        try:
            from db import get_engine, get_session_factory, Repository
            from seed_service import run_demo_seed_pipeline
            engine = get_engine()
            run_demo_seed_pipeline(background=False, engine=engine)
            SessionLocal = get_session_factory(engine)
            session = SessionLocal()
            db_repos = session.query(Repository).all()
            repos_list = [
                {
                    "id": r.id,
                    "url": r.url,
                    "owner": r.owner,
                    "name": r.name,
                    "branch": r.branch,
                    "commit_sha": r.commit_sha,
                    "status": r.status.value if hasattr(r.status, "value") else str(r.status),
                    "total_files": r.total_files or 0,
                    "total_lines": r.total_lines or 0,
                    "total_size_bytes": r.total_size_bytes or 0,
                    "runnability_score": r.runnability_score.value if hasattr(r.runnability_score, "value") else str(r.runnability_score),
                    "is_demo": r.is_demo,
                }
                for r in db_repos
            ]
            session.close()
        except Exception:
            repos_list = []

    indexed_repos_app = [r for r in repos_list if r.get("status") == "INDEXED"] if isinstance(repos_list, list) else []
    pending_repos_app = [r for r in repos_list if r.get("status") != "INDEXED"] if isinstance(repos_list, list) else []

    demo_repos = [r for r in indexed_repos_app if r.get("is_demo") or r.get("owner") in ["encode", "pallets", "psf"]]
    user_repos = [r for r in indexed_repos_app if not r.get("is_demo") and r.get("owner") not in ["encode", "pallets", "psf"]]

    # If demo_repos is empty, ensure the 3 curated demo repositories are populated
    if not demo_repos:
        try:
            from seed_service import run_demo_seed_pipeline
            from db import get_engine
            run_demo_seed_pipeline(background=False, engine=get_engine())
            repos_list = fetch_api("/api/repositories")
            if isinstance(repos_list, list):
                indexed_repos_app = [r for r in repos_list if r.get("status") == "INDEXED"]
                demo_repos = [r for r in indexed_repos_app if r.get("is_demo") or r.get("owner") in ["encode", "pallets", "psf"]]
                user_repos = [r for r in indexed_repos_app if not r.get("is_demo") and r.get("owner") not in ["encode", "pallets", "psf"]]
        except Exception:
            pass

    # Default selection to first demo repository if not set
    if not st.session_state.selected_repo_id and demo_repos:
        st.session_state.selected_repo_id = demo_repos[0]["id"]

    # --- SECTION 2: PRE-INDEXED DEMO REPOSITORIES DROPDOWN ---
    st.markdown("---")
    st.markdown("### ⭐ Demo Repositories")
    if demo_repos:
        demo_options = {f"⭐ {r['owner']}/{r['name']} ({r.get('total_files', 0)} files)": r["id"] for r in demo_repos}
        demo_labels = list(demo_options.keys())

        demo_active_idx = 0
        is_demo_active = False
        for idx, r_id in enumerate(demo_options.values()):
            if r_id == st.session_state.selected_repo_id:
                demo_active_idx = idx
                is_demo_active = True
                break

        def on_demo_change():
            selected_label = st.session_state.demo_repo_selector
            st.session_state.selected_repo_id = demo_options[selected_label]
            st.session_state.chat_history = []

        selected_demo_label = st.selectbox(
            "Select Demo Repository:",
            options=demo_labels,
            index=demo_active_idx,
            key="demo_repo_selector",
            on_change=on_demo_change,
            help="Curated pre-indexed repositories ready for instant grounded Q&A.",
        )
        
        chosen_demo_id = demo_options[selected_demo_label]
        if st.session_state.selected_repo_id != chosen_demo_id:
            if st.button("👉 Switch to this Demo Repo", key="btn_switch_demo", use_container_width=True):
                st.session_state.selected_repo_id = chosen_demo_id
                st.session_state.chat_history = []
                st.rerun()

        # If currently active repo is a demo repo, show details card
        if is_demo_active:
            active_demo = next((r for r in demo_repos if r["id"] == st.session_state.selected_repo_id), demo_repos[0])
            with st.container(border=True):
                st.markdown(f"**Active:** `{active_demo['owner']}/{active_demo['name']}` ⭐ *(Curated Demo)*")
                st.caption(f"📁 Files: {active_demo.get('total_files', 0)} | 🔖 Commit: `{active_demo.get('commit_sha', '')[:7]}` | 🛡️ **Protected**")
                st.info("🛡️ Pre-indexed demo repository. Grounded Q&A is fully enabled in the main tabs!")

    # --- SECTION 3: USER INGESTED REPOSITORIES DROPDOWN ---
    st.markdown("---")
    st.markdown("### 👤 Ingested Repositories")
    if user_repos:
        user_options = {f"👤 {r['owner']}/{r['name']} ({r.get('total_files', 0)} files)": r["id"] for r in user_repos}
        user_labels = list(user_options.keys())

        user_active_idx = 0
        is_user_active = False
        for idx, r_id in enumerate(user_options.values()):
            if r_id == st.session_state.selected_repo_id:
                user_active_idx = idx
                is_user_active = True
                break

        def on_user_change():
            selected_label = st.session_state.user_repo_selector
            st.session_state.selected_repo_id = user_options[selected_label]
            st.session_state.chat_history = []

        selected_user_label = st.selectbox(
            "Select Ingested Repository:",
            options=user_labels,
            index=user_active_idx,
            key="user_repo_selector",
            on_change=on_user_change,
            help="Custom repositories ingested during your session.",
        )
        
        chosen_user_id = user_options[selected_user_label]
        if st.session_state.selected_repo_id != chosen_user_id:
            if st.button("👉 Switch to this Ingested Repo", key="btn_switch_user", use_container_width=True):
                st.session_state.selected_repo_id = chosen_user_id
                st.session_state.chat_history = []
                st.rerun()

        # If currently active repo is a user repo, show details card and delete button
        if is_user_active:
            active_user = next((r for r in user_repos if r["id"] == st.session_state.selected_repo_id), user_repos[0])
            with st.container(border=True):
                if st.session_state.confirm_del_repo_id != active_user["id"]:
                    st.markdown(f"**Active:** `{active_user['owner']}/{active_user['name']}` 👤 *(User Session)*")
                    st.caption(f"📁 Files: {active_user.get('total_files', 0)} | 🔖 Commit: `{active_user.get('commit_sha', '')[:7]}` | ⚡ Health: `{active_user.get('runnability_score', '')}`")
                    if st.button("🗑️ Delete Ingested Repo", key="btn_del_active_user", use_container_width=True):
                        st.session_state.confirm_del_repo_id = active_user["id"]
                        st.rerun()
                else:
                    st.error(f"⚠️ **Confirm Deletion**\n\nPermanently delete **`{active_user['owner']}/{active_user['name']}`**?")
                    c_yes, c_no = st.columns([1, 1])
                    with c_yes:
                        if st.button("🗑️ Yes, Delete", key="btn_yes_del_user", type="primary", use_container_width=True):
                            fetch_api(f"/api/repositories/{active_user['id']}", method="DELETE")
                            st.session_state.selected_repo_id = demo_repos[0]["id"] if demo_repos else None
                            st.session_state.confirm_del_repo_id = None
                            st.session_state.chat_history = []
                            st.toast("Repository deleted!")
                            st.rerun()
                    with c_no:
                        if st.button("Cancel", key="btn_cancel_del_user", use_container_width=True):
                            st.session_state.confirm_del_repo_id = None
                            st.rerun()

        if st.button("🧹 Clean All User Session Repos", key="btn_clean_all_session", use_container_width=True):
            fetch_api("/api/cleanup-session", method="POST")
            st.session_state.selected_repo_id = demo_repos[0]["id"] if demo_repos else None
            st.session_state.confirm_del_repo_id = None
            st.session_state.chat_history = []
            st.toast("Temporary user session repositories cleaned!")
            st.rerun()
    else:
        st.caption("ℹ️ No custom repositories ingested yet. Enter a GitHub repository link in the query bar above to ingest one.")

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
if st.session_state.selected_repo_id not in indexed_ids and demo_repos:
    st.session_state.selected_repo_id = demo_repos[0]["id"]

if not st.session_state.selected_repo_id:
    st.warning("👈 Please enter a GitHub repository URL in the sidebar to begin ingestion or select an indexed repo.")
else:
    active_repo_id = st.session_state.selected_repo_id
    summary = fetch_api(f"/api/repositories/{active_repo_id}/summary")

    if "error" in summary:
        # If API returned error, attempt direct DB summary
        try:
            from db import get_engine, get_session_factory, Repository
            from main import resolve_index_path
            engine = get_engine()
            SessionLocal = get_session_factory(engine)
            session = SessionLocal()
            r_obj = session.query(Repository).filter(Repository.id == active_repo_id).first()
            if r_obj:
                summary = {
                    "id": r_obj.id,
                    "url": r_obj.url,
                    "owner": r_obj.owner,
                    "name": r_obj.name,
                    "branch": r_obj.branch,
                    "commit_sha": r_obj.commit_sha,
                    "status": r_obj.status.value if hasattr(r_obj.status, "value") else str(r_obj.status),
                    "total_files": r_obj.total_files or 0,
                    "total_lines": r_obj.total_lines or 0,
                    "total_size_bytes": r_obj.total_size_bytes or 0,
                    "language_distribution": r_obj.language_distribution or {},
                    "runnability_score": r_obj.runnability_score.value if hasattr(r_obj.runnability_score, "value") else str(r_obj.runnability_score),
                    "health_details": r_obj.health_details or {},
                    "is_demo": r_obj.is_demo,
                }
            session.close()
        except Exception:
            pass

    if "error" in summary:
        # Gracefully handle stale or deleted repository IDs
        st.session_state.selected_repo_id = demo_repos[0]["id"] if demo_repos else None
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
