import os
import time

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")

st.set_page_config(page_title="Fuzzy RAG — Biomedical QA", layout="wide")
st.title("Fuzzy RAG — Biomedical QA")

tab_ask, tab_data, tab_eval, tab_system = st.tabs(["Ask", "Data", "Evaluate", "System"])


# ── Ask ───────────────────────────────────────────────────────────────────────

with tab_ask:
    settings_col, main_col = st.columns([1, 3])

    with settings_col:
        st.subheader("Settings")
        dataset_filter = st.selectbox(
            "Dataset filter",
            options=["All", "pubmedqa", "medqa", "radqa"],
        )
        top_k = st.slider("Top-K contexts", min_value=1, max_value=20, value=5)

    with main_col:
        st.subheader("Ask a Biomedical Question")
        question = st.text_area(
            "Question",
            height=100,
            placeholder="e.g. What are the symptoms of pneumonia?",
        )

        if st.button("Submit", type="primary", key="ask_submit"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving and generating answer..."):
                    try:
                        payload = {"question": question, "top_k": top_k}
                        if dataset_filter != "All":
                            payload["dataset_filter"] = dataset_filter

                        resp = requests.post(
                            f"{BACKEND_URL}/query", json=payload, timeout=120
                        )

                        if resp.ok:
                            data = resp.json()

                            st.subheader("Answer")
                            st.write(data["answer"])

                            with st.expander(f"Retrieved Contexts ({len(data['contexts'])})"):
                                for i, ctx in enumerate(data["contexts"], start=1):
                                    st.markdown(
                                        f"**[{i}]** `{ctx['source']}` — "
                                        f"RRF score: `{ctx['rrf_score']:.4f}`"
                                    )
                                    st.write(ctx["text"])
                                    st.divider()

                            with st.expander("Usage"):
                                usage = data.get("usage", {})
                                st.write(f"Model: `{usage.get('model')}`")
                                st.write(f"Input tokens: {usage.get('input_tokens')}")
                                st.write(f"Output tokens: {usage.get('output_tokens')}")
                        else:
                            st.error(f"Backend error ({resp.status_code}): {resp.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")


# ── Data ──────────────────────────────────────────────────────────────────────

with tab_data:
    dl_col, idx_col = st.columns(2)

    # ---- Download -----------------------------------------------------------
    with dl_col:
        st.subheader("Download Datasets")
        st.caption(
            "Pre-fetches datasets to local cache without building the index. "
            "RadQA requires PhysioNet credentials set in the backend environment."
        )

        dl_datasets = st.multiselect(
            "Datasets to download",
            options=["pubmedqa", "medqa", "radqa"],
            default=["pubmedqa", "medqa"],
            key="dl_datasets",
        )

        if st.button("Start Download", key="dl_start"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/download",
                    json={"datasets": dl_datasets},
                    timeout=10,
                )
                if resp.ok:
                    st.session_state["dl_polling"] = True
                    st.rerun()
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(str(e))

        if st.session_state.get("dl_polling"):
            try:
                status = requests.get(f"{BACKEND_URL}/download/status", timeout=5).json()
                state = status.get("state", "idle")

                if state == "running":
                    current = status.get("current_dataset") or "..."
                    st.info(f"Downloading **{current}**…")
                    time.sleep(2)
                    st.rerun()
                elif state == "done":
                    st.session_state["dl_polling"] = False
                    st.success("Download complete.")
                elif state == "error":
                    st.session_state["dl_polling"] = False
                    st.error(f"Download failed: {status.get('error')}")
                else:
                    st.session_state["dl_polling"] = False
            except Exception as e:
                st.session_state["dl_polling"] = False
                st.error(str(e))

    # ---- Index --------------------------------------------------------------
    with idx_col:
        st.subheader("Index Datasets")
        st.caption(
            "Embeds and stores documents in PostgreSQL. "
            "Downloads source data automatically if not already cached."
        )

        idx_datasets = st.multiselect(
            "Datasets to index",
            options=["pubmedqa", "medqa", "radqa"],
            default=["pubmedqa", "medqa"],
            key="idx_datasets",
        )

        if st.button("Start Indexing", key="idx_start"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/index",
                    json={"datasets": idx_datasets},
                    timeout=10,
                )
                if resp.ok:
                    st.session_state["idx_polling"] = True
                    st.rerun()
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(str(e))

        if st.session_state.get("idx_polling"):
            try:
                status = requests.get(f"{BACKEND_URL}/index/status", timeout=5).json()
                state = status.get("state", "idle")

                if state == "running":
                    current = status.get("current_dataset") or "..."
                    indexed = status.get("indexed", 0)
                    errors = status.get("errors", 0)
                    st.info(
                        f"Indexing **{current}**… "
                        f"{indexed:,} records indexed"
                        + (f", {errors} errors" if errors else "")
                    )
                    time.sleep(2)
                    st.rerun()
                elif state == "done":
                    st.session_state["idx_polling"] = False
                    total = status.get("indexed", 0)
                    st.success(f"Indexing complete — {total:,} records stored.")
                elif state == "error":
                    st.session_state["idx_polling"] = False
                    st.error("Indexing encountered an error. Check backend logs.")
                else:
                    st.session_state["idx_polling"] = False
            except Exception as e:
                st.session_state["idx_polling"] = False
                st.error(str(e))


# ── Evaluate ──────────────────────────────────────────────────────────────────

with tab_eval:
    st.subheader("Batch Evaluation")

    eval_form_col, eval_result_col = st.columns([2, 1])

    with eval_form_col:
        eval_questions_raw = st.text_area(
            "Questions (one per line)", height=150,
            placeholder="What are the symptoms of pneumonia?\nHow is TB diagnosed?",
        )
        eval_references_raw = st.text_area(
            "Reference answers (one per line)", height=150,
            placeholder="Fever, cough, shortness of breath…\nSputum culture, chest X-ray…",
        )

        eval_filter_col, save_col = st.columns(2)
        with eval_filter_col:
            eval_dataset_filter = st.selectbox(
                "Dataset filter",
                options=["All", "pubmedqa", "medqa", "radqa"],
                key="eval_filter",
            )
        with save_col:
            save_name = st.text_input(
                "Save as baseline (leave blank to skip)",
                placeholder="e.g. medgemma_pubmedqa_v1",
            )

        if st.button("Run Evaluation", type="primary", key="eval_run"):
            eval_questions = [
                q.strip() for q in eval_questions_raw.strip().splitlines() if q.strip()
            ]
            eval_references = [
                r.strip() for r in eval_references_raw.strip().splitlines() if r.strip()
            ]

            if not eval_questions or not eval_references:
                st.warning("Please enter both questions and reference answers.")
            elif len(eval_questions) != len(eval_references):
                st.error("Number of questions and reference answers must match.")
            else:
                with st.spinner(f"Evaluating {len(eval_questions)} question(s)…"):
                    try:
                        payload = {
                            "questions": eval_questions,
                            "references": eval_references,
                        }
                        if eval_dataset_filter != "All":
                            payload["dataset_filter"] = eval_dataset_filter
                        if save_name.strip():
                            payload["save_as"] = save_name.strip()

                        resp = requests.post(
                            f"{BACKEND_URL}/eval", json=payload, timeout=600
                        )

                        if resp.ok:
                            st.session_state["eval_result"] = resp.json()
                            st.rerun()
                        else:
                            st.error(f"Error: {resp.text}")
                    except Exception as e:
                        st.error(str(e))

    with eval_result_col:
        result = st.session_state.get("eval_result")
        if result:
            st.subheader("Results")
            st.metric("BLEU", f"{result['bleu']:.4f}")
            st.metric("ROUGE-L", f"{result['rouge_l']:.4f}")
            st.metric("Token F1", f"{result['token_f1']:.4f}")
            st.caption(f"{result['n_samples']} sample(s) evaluated.")
            if result.get("saved"):
                saved = result["saved"]
                st.success(
                    f"Saved as **{saved['name']}** "
                    f"({saved['timestamp'][:19].replace('T', ' ')} UTC)"
                )

    # ---- Saved baselines ----------------------------------------------------
    st.divider()
    st.subheader("Saved Baselines")

    refresh_col, _ = st.columns([1, 5])
    with refresh_col:
        if st.button("Refresh", key="baselines_refresh"):
            st.rerun()

    try:
        baselines = requests.get(f"{BACKEND_URL}/eval/baselines", timeout=5).json()
        if not baselines:
            st.info("No baselines saved yet. Run an evaluation with a name to save one.")
        else:
            rows = []
            for b in baselines:
                m = b.get("metrics", {})
                rows.append({
                    "Name": b["name"],
                    "Timestamp (UTC)": b["timestamp"][:19].replace("T", " "),
                    "Model": b.get("model", ""),
                    "Dataset": b.get("dataset_filter") or "All",
                    "BLEU": m.get("bleu"),
                    "ROUGE-L": m.get("rouge_l"),
                    "Token F1": m.get("token_f1"),
                    "Samples": m.get("n_samples"),
                })
            st.dataframe(rows, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load baselines: {e}")


# ── System ────────────────────────────────────────────────────────────────────

with tab_system:
    st.subheader("Backend Health")

    if st.button("Check Health", key="health_check"):
        try:
            resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if resp.ok:
                data = resp.json()
                st.success(f"Status: **{data['status']}**")
                st.write(f"Model: `{data.get('model')}`")
            else:
                st.error(f"Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.subheader("Index Status")

    if st.button("Check Index Status", key="idx_status_check"):
        try:
            resp = requests.get(f"{BACKEND_URL}/index/status", timeout=5)
            if resp.ok:
                st.json(resp.json())
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.subheader("Download Status")

    if st.button("Check Download Status", key="dl_status_check"):
        try:
            resp = requests.get(f"{BACKEND_URL}/download/status", timeout=5)
            if resp.ok:
                st.json(resp.json())
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(str(e))
