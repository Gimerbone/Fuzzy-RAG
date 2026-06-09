import os
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")

st.set_page_config(page_title="Fuzzy RAG — Biomedical QA", layout="wide")
st.title("Fuzzy RAG — Biomedical QA")

# Sidebar
with st.sidebar:
    st.header("Settings")
    dataset_filter = st.selectbox(
        "Dataset filter",
        options=["All", "pubmedqa", "medqa", "radqa"],
    )
    top_k = st.slider("Top-K contexts", min_value=1, max_value=20, value=5)

    st.divider()
    st.header("Index Datasets")
    selected_datasets = st.multiselect(
        "Datasets to index",
        options=["pubmedqa", "medqa", "radqa"],
        default=["pubmedqa", "medqa"],
    )
    if st.button("Start Indexing"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/index",
                json={"datasets": selected_datasets},
                timeout=10,
            )
            if resp.ok:
                st.success("Indexing started in background.")
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(str(e))

    if st.button("Check Index Status"):
        try:
            resp = requests.get(f"{BACKEND_URL}/index/status", timeout=5)
            if resp.ok:
                status = resp.json()
                st.json(status)
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.header("Backend Health")
    if st.button("Check Health"):
        try:
            resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if resp.ok:
                st.json(resp.json())
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(str(e))

# Main QA interface
st.subheader("Ask a Biomedical Question")
question = st.text_area("Question", height=100, placeholder="e.g. What are the symptoms of pneumonia?")

if st.button("Submit", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                payload = {
                    "question": question,
                    "top_k": top_k,
                }
                if dataset_filter != "All":
                    payload["dataset_filter"] = dataset_filter

                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json=payload,
                    timeout=120,
                )

                if resp.ok:
                    data = resp.json()

                    st.subheader("Answer")
                    st.write(data["answer"])

                    with st.expander(f"Retrieved Contexts ({len(data['contexts'])})"):
                        for i, ctx in enumerate(data["contexts"], start=1):
                            st.markdown(
                                f"**[{i}]** `{ctx['source']}` — RRF score: `{ctx['rrf_score']:.4f}`"
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

# Evaluation section
st.divider()
st.subheader("Batch Evaluation (BLEU / ROUGE-L / F1)")

with st.expander("Run Evaluation"):
    st.markdown(
        "Provide one question per line, and the corresponding reference answer per line. "
        "The system will retrieve + generate for each question and score against the references."
    )
    eval_questions_raw = st.text_area("Questions (one per line)", height=120)
    eval_references_raw = st.text_area("Reference answers (one per line)", height=120)

    if st.button("Evaluate"):
        eval_questions = [q.strip() for q in eval_questions_raw.strip().splitlines() if q.strip()]
        eval_references = [r.strip() for r in eval_references_raw.strip().splitlines() if r.strip()]

        if not eval_questions or not eval_references:
            st.warning("Please enter both questions and reference answers.")
        elif len(eval_questions) != len(eval_references):
            st.error("Number of questions and reference answers must match.")
        else:
            with st.spinner("Evaluating..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/eval",
                        json={"questions": eval_questions, "references": eval_references},
                        timeout=300,
                    )
                    if resp.ok:
                        metrics = resp.json()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("BLEU", f"{metrics['bleu']:.4f}")
                        col2.metric("ROUGE-L", f"{metrics['rouge_l']:.4f}")
                        col3.metric("Token F1", f"{metrics['token_f1']:.4f}")
                        st.caption(f"Evaluated on {metrics['n_samples']} samples.")
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(str(e))
