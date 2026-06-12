import base64
import os
import time

import plotly.graph_objects as go
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

        uploaded_image = st.file_uploader(
            "Upload image (optional)",
            type=["jpg", "jpeg", "png"],
            help=(
                "Attach a radiograph, pathology slide, or any biomedical image. "
                "BiomedCLIP will encode it for cross-modal retrieval against the text index, "
                "and MedGemma will incorporate it into the answer."
            ),
        )
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded image", use_container_width=False, width=300)

        if st.button("Submit", type="primary", key="ask_submit"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving and generating answer..."):
                    try:
                        payload: dict = {
                            "question": question,
                            "top_k": top_k,
                        }
                        if dataset_filter != "All":
                            payload["dataset_filter"] = dataset_filter
                        if uploaded_image:
                            uploaded_image.seek(0)
                            payload["image"] = base64.b64encode(uploaded_image.read()).decode()

                        resp = requests.post(
                            f"{BACKEND_URL}/query", json=payload, timeout=600
                        )

                        if resp.ok:
                            data = resp.json()

                            if data.get("image_used"):
                                st.caption("Image used for cross-modal retrieval + generation.")

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
            "PubMedQA and MedQA are already cached if previously downloaded. "
            "RadQA requires PhysioNet credentials set in the backend environment."
        )

        dl_datasets = st.multiselect(
            "Datasets to download",
            options=["pubmedqa", "medqa", "radqa"],
            default=["pubmedqa", "medqa"],
            key="dl_datasets",
            help="PubMedQA/MedQA use HuggingFace's local cache — re-running is safe but a no-op. RadQA skips files already on disk.",
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

        st.divider()
        with st.expander("Manual RadQA upload (if automatic download fails)"):
            st.caption(
                "Download the SQuAD-format JSON files from "
                "[PhysioNet radqa/1.0.0](https://physionet.org/content/radqa/1.0.0/) "
                "and upload them here. At minimum the train split is required."
            )
            up_train = st.file_uploader("Train split (train.json)", type=["json"], key="up_radqa_train")
            up_dev   = st.file_uploader("Dev split (dev.json)",   type=["json"], key="up_radqa_dev")
            up_test  = st.file_uploader("Test split (test.json)", type=["json"], key="up_radqa_test")

            if st.button("Upload", key="radqa_upload_btn", disabled=not any([up_train, up_dev, up_test])):
                files = {}
                if up_train:
                    files["train"] = (up_train.name, up_train.getvalue(), "application/json")
                if up_dev:
                    files["dev"]   = (up_dev.name,   up_dev.getvalue(),   "application/json")
                if up_test:
                    files["test"]  = (up_test.name,  up_test.getvalue(),  "application/json")

                with st.spinner("Uploading…"):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/upload/radqa", files=files, timeout=120
                        )
                        if resp.ok:
                            result = resp.json()
                            if result.get("saved"):
                                st.success(f"Saved: {', '.join(result['saved'])}")
                            for err in result.get("errors", []):
                                st.error(err)
                        else:
                            st.error(f"Upload failed ({resp.status_code}): {resp.text}")
                    except Exception as e:
                        st.error(str(e))

    # ---- Index --------------------------------------------------------------
    with idx_col:
        st.subheader("Index Datasets")
        st.caption(
            "Embeds and stores documents in PostgreSQL. "
            "Downloads source data automatically if not already cached. "
            "Datasets already in the index are skipped automatically."
        )

        # Fetch per-dataset record counts to show what's already indexed
        try:
            source_counts = requests.get(f"{BACKEND_URL}/index/stats", timeout=5).json()
        except Exception:
            source_counts = {}

        all_datasets = ["pubmedqa", "medqa", "radqa"]
        for ds in all_datasets:
            n = source_counts.get(ds, 0)
            if n > 0:
                st.success(f"**{ds}** — {n:,} records indexed")
            else:
                st.warning(f"**{ds}** — not indexed yet")

        not_indexed = [ds for ds in all_datasets if source_counts.get(ds, 0) == 0]

        idx_datasets = st.multiselect(
            "Datasets to index",
            options=all_datasets,
            default=not_indexed,
            key="idx_datasets",
            help="Datasets already in the index will be skipped by the backend.",
        )

        already_selected = [ds for ds in idx_datasets if source_counts.get(ds, 0) > 0]
        if already_selected:
            st.info(
                f"{', '.join(f'**{d}**' for d in already_selected)} "
                f"{'is' if len(already_selected) == 1 else 'are'} already indexed "
                f"and will be skipped."
            )

        if st.button("Start Indexing", key="idx_start", disabled=not idx_datasets):
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
                    st.rerun()
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
    eval_single_tab, eval_compare_tab, eval_baselines_tab = st.tabs(
        ["Single Run", "Fuzzy vs Standard RAG", "Saved Baselines"]
    )

    # ── Single Run ────────────────────────────────────────────────────────────
    with eval_single_tab:
        st.subheader("Batch Evaluation")
        st.caption(
            "Run the full Fuzzy RAG pipeline and measure BLEU, ROUGE-L, and Token F1 "
            "against your reference answers."
        )

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

                # Mini bar chart for single-run metrics
                fig = go.Figure(go.Bar(
                    x=["BLEU", "ROUGE-L", "Token F1"],
                    y=[result["bleu"], result["rouge_l"], result["token_f1"]],
                    marker_color=["#4C78A8", "#72B7B2", "#54A24B"],
                    text=[f"{v:.4f}" for v in [result["bleu"], result["rouge_l"], result["token_f1"]]],
                    textposition="outside",
                ))
                fig.update_layout(
                    yaxis=dict(range=[0, 1], title="Score"),
                    margin=dict(t=20, b=20),
                    height=280,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Fuzzy vs Standard RAG ─────────────────────────────────────────────────
    with eval_compare_tab:
        st.subheader("Fuzzy RAG vs Standard RAG")
        st.caption(
            "Runs the same questions through two pipelines and compares generation quality. "
            "**Fuzzy RAG** uses dense + BM25 + fuzzy metadata retrieval fused with RRF. "
            "**Standard RAG** uses dense + BM25 only. The delta shows the fuzzy metadata effect."
        )

        cmp_q_raw = st.text_area(
            "Questions (one per line)", height=130,
            placeholder="What are the symptoms of pneumonia?\nHow is TB diagnosed?",
            key="cmp_q",
        )
        cmp_r_raw = st.text_area(
            "Reference answers (one per line)", height=130,
            placeholder="Fever, cough, shortness of breath…\nSputum culture, chest X-ray…",
            key="cmp_r",
        )
        cmp_filter = st.selectbox(
            "Dataset filter",
            options=["All", "pubmedqa", "medqa", "radqa"],
            key="cmp_filter",
        )

        if st.button("Run Comparison", type="primary", key="cmp_run"):
            cmp_questions = [q.strip() for q in cmp_q_raw.strip().splitlines() if q.strip()]
            cmp_references = [r.strip() for r in cmp_r_raw.strip().splitlines() if r.strip()]

            if not cmp_questions or not cmp_references:
                st.warning("Please enter both questions and reference answers.")
            elif len(cmp_questions) != len(cmp_references):
                st.error("Number of questions and reference answers must match.")
            else:
                with st.spinner(
                    f"Running both pipelines on {len(cmp_questions)} question(s) — this takes a while…"
                ):
                    try:
                        payload = {
                            "questions": cmp_questions,
                            "references": cmp_references,
                        }
                        if cmp_filter != "All":
                            payload["dataset_filter"] = cmp_filter

                        resp = requests.post(
                            f"{BACKEND_URL}/eval/compare", json=payload, timeout=1200
                        )
                        if resp.ok:
                            st.session_state["cmp_result"] = resp.json()
                            st.rerun()
                        else:
                            st.error(f"Error: {resp.text}")
                    except Exception as e:
                        st.error(str(e))

        cmp = st.session_state.get("cmp_result")
        if cmp:
            fuzzy_m = cmp["fuzzy_rag"]
            std_m = cmp["standard_rag"]
            delta = cmp["delta"]

            st.divider()
            st.subheader(f"Results — {cmp['n_samples']} sample(s)")

            # Metric cards with delta
            mc1, mc2, mc3 = st.columns(3)
            for col, metric, label in [
                (mc1, "bleu", "BLEU"),
                (mc2, "rouge_l", "ROUGE-L"),
                (mc3, "token_f1", "Token F1"),
            ]:
                d = delta[metric]
                delta_str = f"+{d:.4f}" if d >= 0 else f"{d:.4f}"
                col.metric(
                    label=f"{label} (Fuzzy RAG)",
                    value=f"{fuzzy_m[metric]:.4f}",
                    delta=delta_str,
                    help=f"Standard RAG: {std_m[metric]:.4f}",
                )

            # Grouped bar chart
            metrics_labels = ["BLEU", "ROUGE-L", "Token F1"]
            fuzzy_vals = [fuzzy_m["bleu"], fuzzy_m["rouge_l"], fuzzy_m["token_f1"]]
            std_vals = [std_m["bleu"], std_m["rouge_l"], std_m["token_f1"]]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Fuzzy RAG",
                x=metrics_labels,
                y=fuzzy_vals,
                marker_color="#4C78A8",
                text=[f"{v:.4f}" for v in fuzzy_vals],
                textposition="outside",
            ))
            fig.add_trace(go.Bar(
                name="Standard RAG",
                x=metrics_labels,
                y=std_vals,
                marker_color="#F58518",
                text=[f"{v:.4f}" for v in std_vals],
                textposition="outside",
            ))
            fig.update_layout(
                barmode="group",
                yaxis=dict(range=[0, 1], title="Score"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=30, b=20),
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Delta bar chart (positive = fuzzy wins)
            delta_vals = [delta["bleu"], delta["rouge_l"], delta["token_f1"]]
            colors = ["#54A24B" if v >= 0 else "#E45756" for v in delta_vals]
            fig2 = go.Figure(go.Bar(
                x=metrics_labels,
                y=delta_vals,
                marker_color=colors,
                text=[f"{v:+.4f}" for v in delta_vals],
                textposition="outside",
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(
                title="Delta: Fuzzy RAG minus Standard RAG (green = fuzzy wins)",
                yaxis_title="Score difference",
                margin=dict(t=40, b=20),
                height=300,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Saved Baselines ───────────────────────────────────────────────────────
    with eval_baselines_tab:
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
    st.subheader("Model Health")
    st.caption(
        "Both BiomedCLIP and MedGemma are **pretrained models** that require no training before use. "
        "They are downloaded from HuggingFace and ready immediately. "
        "Clicking the button below runs a real inference test. The **first** call loads "
        "both models into GPU memory and can take a few minutes; later calls return in "
        "well under a second."
    )

    if st.button("Check Model Health", key="model_health_btn"):
        with st.spinner(
            "Running inference tests on both models… the first call can take several "
            "minutes while BiomedCLIP and MedGemma load into GPU memory."
        ):
            try:
                resp = requests.get(f"{BACKEND_URL}/health/models", timeout=600)
                if resp.ok:
                    st.session_state["model_health"] = resp.json()
                else:
                    st.error(f"Error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(str(e))

    health = st.session_state.get("model_health")
    if health:
        emb = health.get("embedding", {})
        llm = health.get("llm", {})

        h_col1, h_col2 = st.columns(2)

        with h_col1:
            ok = emb.get("status") == "ok"
            st.markdown(f"### {'✅' if ok else '❌'} BiomedCLIP")
            st.caption("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            if ok:
                st.success("Running")
                st.write(f"**Device:** `{emb.get('device')}`")
                st.write(f"**Embedding dim:** `{emb.get('output_dim')}`")
                st.write(f"**Latency:** `{emb.get('latency_ms')} ms`")
            else:
                st.error(emb.get("error", "Unknown error"))

        with h_col2:
            ok = llm.get("status") == "ok"
            st.markdown(f"### {'✅' if ok else '❌'} MedGemma")
            st.caption("google/medgemma-4b-it")
            if ok:
                st.success("Running")
                st.write(f"**Device:** `{llm.get('device')}`")
                st.write(f"**Quantization:** `{llm.get('quantization')}`")
                st.write(f"**Latency:** `{llm.get('latency_ms')} ms`")
            else:
                st.error(llm.get("error", "Unknown error"))

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
