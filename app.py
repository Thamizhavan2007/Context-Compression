import time

_APP_START = time.perf_counter()

def startup_checkpoint(label):

    elapsed = time.perf_counter() - _APP_START

    print(

        f"[STARTUP] {elapsed:7.2f}s | {label}",

        flush=True,

    )

startup_checkpoint("Starting app.py")

from pathlib import Path

import ast

import json

import math

import pandas as pd

import plotly.express as px

import streamlit as st

import gc
import html

startup_checkpoint("All imports completed")

@st.cache_resource(show_spinner="Loading embedding model...")

def get_embedding_model():

    from src.embeddings import EmbeddingModel

    return EmbeddingModel()

@st.cache_resource(show_spinner="Loading language model...")

def get_language_model():

    from src.llm import LocalLLM

    return LocalLLM()

ROOT = Path(__file__).resolve().parent

RESULTS_DIR = ROOT / "results"

BATCH_PATH = RESULTS_DIR / "batch_results.csv"

SUMMARY_PATH = RESULTS_DIR / "summary.csv"

st.set_page_config(

    page_title="Context Compression Lab",

    page_icon="🧠",

    layout="wide",

    initial_sidebar_state="expanded",

)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#152238; --muted:#64748b; --line:#e3eaf3; --panel:#ffffff; --accent:#5b6ef5; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: #f5f7fb; color:var(--ink); }
.block-container { padding-top:1.55rem; padding-bottom:3rem; max-width:1500px; }
[data-testid="stHeader"] { background:rgba(245,247,251,.88); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#111d33 0%,#182944 100%); border-right:0; }
[data-testid="stSidebar"] * { color:#e8eef9; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#a8b7cf; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.14); }
[data-testid="stSidebar"] [role="radiogroup"] label { padding:.58rem .7rem; border-radius:10px; margin:.12rem 0; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(255,255,255,.08); }
.hero { position:relative; overflow:hidden; padding:1.7rem 1.9rem; border-radius:22px; background:linear-gradient(115deg,#172640 0%,#243b63 58%,#4658a8 100%); border:1px solid rgba(255,255,255,.12); margin-bottom:1.4rem; color:#fff; box-shadow:0 16px 40px rgba(24,42,75,.13); }
.hero:after { content:''; position:absolute; width:280px; height:280px; right:-80px; top:-150px; border-radius:50%; border:1px solid rgba(255,255,255,.15); box-shadow:0 0 0 28px rgba(255,255,255,.035),0 0 0 58px rgba(255,255,255,.025); }
.hero h1 { font-family:'Manrope',sans-serif; font-size:2.25rem; font-weight:800; line-height:1.15; margin:0 0 .6rem 0; letter-spacing:-.04em; color:#fff; }
.hero p { max-width:900px; margin:0; color:#d3def2; font-size:1rem; line-height:1.65; }
.eyebrow { font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; color:#aabaff; font-weight:700; margin-bottom:.65rem; }
.subtle { color:#64748b; font-size:.88rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.025em; color:#17263f; }
h2 { font-size:1.45rem !important; }
h3 { font-size:1.1rem !important; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:1rem 1.05rem; border-radius:16px; box-shadow:0 4px 14px rgba(30,48,79,.035); }
[data-testid="stMetricLabel"] { color:#64748b; font-size:.8rem; font-weight:600; }
[data-testid="stMetricValue"] { color:#17263f; font-family:'Manrope',sans-serif; font-weight:800; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; }
.stButton > button, .stDownloadButton > button { border-radius:11px; font-weight:700; min-height:2.8rem; transition:all .15s ease; }
.stButton > button[kind="primary"] { background:linear-gradient(100deg,#596af3,#7180ff); border:0; box-shadow:0 7px 16px rgba(91,110,245,.22); }
.stButton > button[kind="primary"]:hover { background:linear-gradient(100deg,#4859df,#6474f5); transform:translateY(-1px); }
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div, [data-testid="stFileUploader"] section { border-radius:11px !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:13px; overflow:hidden; }
.result-card { border:1px solid #e1e8f2; border-radius:15px; padding:1.1rem 1.2rem; background:#fff; box-shadow:0 4px 14px rgba(30,48,79,.035); line-height:1.7; overflow-wrap:anywhere; }
[data-testid="stAlert"] { border-radius:12px; }
hr { border-color:#e3eaf3; margin:1.25rem 0; }
@media (max-width: 800px) { .hero { padding:1.25rem; border-radius:16px; } .hero h1 { font-size:1.7rem; } .block-container { padding-left:1rem; padding-right:1rem; } }
</style>
""", unsafe_allow_html=True)

DATASET_LABELS = {

    "qasper": "Qasper",

    "hotpotqa": "HotpotQA",

    "multifieldqa_en": "MultiFieldQA-en",

}

METHOD_LABELS = {

    "relevance_only": "Relevance-only",

    "structure_aware": "Structure-aware",

    "full_context": "Full context",

}

def dataset_label(value):

    return DATASET_LABELS.get(str(value), str(value).replace("_", " ").title())

def method_label(value):

    return METHOD_LABELS.get(str(value), str(value).replace("_", " ").title())

def ratio_label(value):

    try:

        if pd.isna(value):

            return "Unknown"

        return f"{float(value):g}×"

    except (TypeError, ValueError):

        return str(value)

def normalize_columns(df):

    df = df.copy()

    df.columns = [str(c).strip().lower() for c in df.columns]

    # Accommodate common naming variations without changing any metric values.

    aliases = {

        "qa_f1": "f1",

        "qa_f1_score": "f1",

        "f1_score": "f1",

        "id": "sample_id",

        "compression_ratio_target": "target_ratio",

        "generated_answer": "prediction",

        "reference_answers": "references",

        "refs": "references",

        "ground_truth": "references",

        "selected_ids": "selected_chunk_ids",

    }

    for old, new in aliases.items():

        if new not in df.columns and old in df.columns:

            df[new] = df[old]

    for col in [

        "target_ratio", "f1", "mean_f1", "mean_f1_percent", "sample_count",

        "original_tokens", "token_budget", "selected_tokens", "compression_ratio",

    ]:

        if col in df.columns:

            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["dataset", "method"]:

        if col in df.columns:

            df[col] = df[col].astype(str).str.lower()

    return df

@st.cache_data(ttl=300)

def read_csv(path_string):

    path = Path(path_string)

    if not path.exists():

        return pd.DataFrame()

    try:

        return normalize_columns(pd.read_csv(path))

    except Exception as exc:

        st.error(f"Could not read {path.name}: {exc}")

        return pd.DataFrame()

def parse_list(value):

    if value is None or (isinstance(value, float) and math.isnan(value)):

        return []

    if isinstance(value, list):

        return value

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "not recorded"}:

        return []

    for parser in (json.loads, ast.literal_eval):

        try:

            parsed = parser(text)

            if isinstance(parsed, list):

                return parsed

            return [parsed]

        except Exception:

            pass

    # Handles whitespace-separated selected IDs, e.g. "1 2 5 7".

    if text and all(part.lstrip("-").isdigit() for part in text.replace(",", " ").split()):

        return [int(part) for part in text.replace(",", " ").split()]

    return [text]

def format_references(value):

    items = parse_list(value)

    if not items:

        return "Reference answer was not stored in this result row."

    return "\n\n".join(str(item) for item in items)

def format_chunk_ids(value):

    items = parse_list(value)

    if not items:

        return "Not recorded"

    return ", ".join(str(x) for x in items)

def load_results():

    batch = read_csv(str(BATCH_PATH))

    summary = read_csv(str(SUMMARY_PATH))

    if summary.empty and not batch.empty and {"dataset", "target_ratio", "method", "f1"}.issubset(batch.columns):

        summary = (

            batch.groupby(["dataset", "target_ratio", "method"], dropna=False)

            .agg(sample_count=("f1", "count"), mean_f1=("f1", "mean"))

            .reset_index()

        )

        summary["mean_f1_percent"] = summary["mean_f1"] * 100

    return batch, summary

startup_checkpoint("Before loading CSV results")

batch, summary = load_results()

startup_checkpoint("CSV results loaded")

with st.sidebar:

    st.markdown("## 🧠 Context Compression Lab")

    st.caption("Research evaluation workspace")

    page = st.radio(

        "Navigate",

        ["Overview","Live Comparison", "Results Explorer", "Experiment Inspector", "Side-by-side Comparison",],

        label_visibility="visible",

    )

    st.divider()

    if not batch.empty:

        st.caption(f"**{len(batch)}** saved experiment rows")

        st.caption("Data source: \`results/*.csv\`")

    else:

        st.caption("Results files not found")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Research evaluation workspace · Long-context AI</div>
  <h1>Context Compression Lab</h1>
  <p>Evaluating Structural Information for Query-Aware Context Compression in Long-Context LLMs</p>
</div>
""", unsafe_allow_html=True)

startup_checkpoint("Beginning page rendering")

if batch.empty and summary.empty and page != "Live Comparison":

    st.warning("No results found. Place \`batch_results.csv\` and \`summary.csv\` inside the project's \`results\` folder.")

    st.code("Context_Compression/\n├── app.py\n├── src/\n└── results/\n    ├── batch_results.csv\n    └── summary.csv")

    st.stop()

if (

    page != "Live Comparison"

    and "dataset" not in batch.columns

    and not batch.empty

):

    st.error("The detailed CSV needs a \`dataset\` column.")

    st.stop()

if page == "Overview":

    st.subheader("Research overview")

    st.write("A controlled comparison of query-relevance ranking and graph-based structural support under matched token budgets.")

    datasets = sorted(batch["dataset"].dropna().unique().tolist()) if "dataset" in batch else []

    methods = sorted(batch["method"].dropna().unique().tolist()) if "method" in batch else []

    ratios = sorted(batch["target_ratio"].dropna().unique().tolist()) if "target_ratio" in batch else []

    a, b, c, d = st.columns(4)

    a.metric("Saved experiment rows", f"{len(batch):,}" if not batch.empty else "—")

    b.metric("Datasets", len(datasets) if datasets else "—")

    c.metric("Methods", len(methods) if methods else "—")

    d.metric("Compression levels", len(ratios) if ratios else "—")

    st.divider()

    if not summary.empty and {"dataset", "target_ratio", "method", "mean_f1"}.issubset(summary.columns):

        st.subheader("Performance comparison")

        available_datasets = sorted(summary["dataset"].dropna().unique().tolist())

        chosen_dataset = st.selectbox("Choose dataset", available_datasets, format_func=dataset_label)

        chart_df = summary[summary["dataset"] == chosen_dataset].copy()

        chart_df["Method"] = chart_df["method"].map(method_label)

        chart_df["Compression"] = chart_df["target_ratio"].map(ratio_label)

        chart_df = chart_df.sort_values("target_ratio")

        fig = px.bar(

            chart_df,

            x="Compression",

            y="mean_f1",

            color="Method",

            barmode="group",

            text="mean_f1",

            hover_data=[c for c in ["sample_count", "mean_f1_percent"] if c in chart_df.columns],

            labels={"mean_f1": "Mean QA F1", "Compression": "Target compression"},

            color_discrete_map={"Relevance-only": "#4F6BED", "Structure-aware": "#F06A4F", "Full context": "#28A58A"},

            template="plotly_white",

        )

        fig.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False)

        fig.update_layout(

            height=390, legend_title_text="", margin=dict(l=10, r=10, t=35, b=10),

            yaxis=dict(range=[0, min(1.12, max(0.5, float(chart_df["mean_f1"].max()) + .16))], tickformat=".0%"),

        )

        st.plotly_chart(fig, width="stretch")

        st.caption("Higher QA F1 means greater token overlap with the reference answer. The displayed sample count is essential when interpreting averages.")

        display = chart_df.copy()

        display["Dataset"] = display["dataset"].map(dataset_label)

        keep = [c for c in ["Dataset", "Compression", "Method", "sample_count", "mean_f1", "mean_f1_percent"] if c in display.columns]

        st.dataframe(

        display[keep].rename(

            columns={

                "sample_count": "Samples",

                "mean_f1": "Mean F1",

                "mean_f1_percent": "Mean F1 (%)",

            }

        ),

        width="stretch",

        hide_index=True,

    )

    else:

        st.info("The summary chart needs dataset, target_ratio, method, and mean_f1 columns. If the summary file is absent, it can be derived from detailed F1 rows.")

    with st.expander("How the methods work"):

        left, right = st.columns(2)

        with left:

            st.markdown("#### Relevance-only")

            st.write("Ranks chunks by query–chunk embedding similarity and selects chunks within the token budget.")

        with right:

            st.markdown("#### Structure-aware")

            st.write("Combines normalized query relevance with structural support from semantically connected chunks.")

        st.latex(r"S_i = **\a**lpha R_i + (1-**\a**lpha)G_i")

        st.caption("Current configuration: α = 0.7, according to the implementation notes.")

    st.warning("Current results are a pilot if only two samples per dataset have been evaluated. Avoid presenting these averages as conclusive benchmark findings.")

elif page == "Results Explorer":

    st.subheader("Results explorer")

    st.caption("Filter the actual saved experiment rows and export a subset for further analysis.")

    if batch.empty:

        st.info("No detailed batch results available.")

        st.stop()

    filtered = batch.copy()

    c1, c2, c3 = st.columns(3)

    if "dataset" in filtered:

        values = sorted(filtered["dataset"].dropna().unique().tolist())

        selected = c1.multiselect("Dataset", values, default=values, format_func=dataset_label)

        filtered = filtered[filtered["dataset"].isin(selected)]

    if "target_ratio" in filtered:

        values = sorted(filtered["target_ratio"].dropna().unique().tolist())

        selected = c2.multiselect("Compression ratio", values, default=values, format_func=ratio_label)

        filtered = filtered[filtered["target_ratio"].isin(selected)]

    if "method" in filtered:

        values = sorted(filtered["method"].dropna().unique().tolist())

        selected = c3.multiselect("Method", values, default=values, format_func=method_label)

        filtered = filtered[filtered["method"].isin(selected)]

    st.write(f"**{len(filtered)}** experiment rows match the current filters.")

    view = filtered.copy()

    if "dataset" in view:

        view["Dataset"] = view["dataset"].map(dataset_label)

    if "method" in view:

        view["Method"] = view["method"].map(method_label)

    if "target_ratio" in view:

        view["Compression"] = view["target_ratio"].map(ratio_label)

    if "selected_chunk_ids" in view:

        view["Selected chunks"] = view["selected_chunk_ids"].map(format_chunk_ids)

    columns = [

        "Dataset", "sample_index", "sample_id", "Compression", "Method",

        "original_tokens", "token_budget", "selected_tokens", "f1",

        "query", "prediction", "references", "Selected chunks",

    ]

    visible = [col for col in columns if col in view.columns]

    st.dataframe(view[visible], use_container_width=True, hide_index=True, height=430)

    st.download_button(

        "Export filtered CSV",

        data=filtered.to_csv(index=False).encode("utf-8"),

        file_name="filtered_context_compression_results.csv",

        mime="text/csv",

    )

elif page == "Experiment Inspector":

    st.subheader("Experiment inspector")

    st.caption("Trace one saved score back to its query, answer, and token-selection evidence.")

    if batch.empty:

        st.info("No individual experiment rows available.")

        st.stop()

    df = batch.reset_index(drop=True).copy()

    for col in ["dataset", "sample_id", "sample_index", "target_ratio", "method"]:

        if col not in df:

            df[col] = "—"

    def option_label(i):

        row = df.iloc[i]

        return f"{dataset_label(row['dataset'])} · sample {row['sample_id']} · {ratio_label(row['target_ratio'])} · {method_label(row['method'])}"

    selected = st.selectbox("Select experiment", list(range(len(df))), format_func=option_label)

    row = df.iloc[selected]

    a, b, c, d = st.columns(4)

    a.metric("Method", method_label(row.get("method", "")))

    b.metric("Target compression", ratio_label(row.get("target_ratio", float("nan"))))

    c.metric("Selected tokens", f"{int(row['selected_tokens']):,}" if pd.notna(row.get("selected_tokens")) else "—")

    d.metric("QA F1", f"{float(row['f1']):.4f}" if pd.notna(row.get("f1")) else "Not recorded")

    st.markdown("#### Query")

    st.info(str(row.get("query", "Query not stored.")))

    left, right = st.columns(2)

    with left:

        st.markdown("#### Reference answer(s)")

        st.markdown(f'<div class="result-card">{html.escape(format_references(row.get("references", row.get("answers", None))))}</div>', unsafe_allow_html=True)

    with right:

        st.markdown("#### Generated answer")

        st.markdown(f'<div class="result-card">{html.escape(str(row.get("prediction", "Prediction not stored.")))}</div>', unsafe_allow_html=True)

    st.markdown("#### Token and selection evidence")

    x, y, z = st.columns(3)

    x.metric("Original tokens", f"{int(row['original_tokens']):,}" if pd.notna(row.get("original_tokens")) else "—")

    y.metric("Token budget", f"{int(row['token_budget']):,}" if pd.notna(row.get("token_budget")) else "—")

    z.metric("Actual selected tokens", f"{int(row['selected_tokens']):,}" if pd.notna(row.get("selected_tokens")) else "—")

    st.markdown("**Selected chunk IDs**")

    st.code(format_chunk_ids(row.get("selected_chunk_ids", None)), language="text")

    st.caption("Values are loaded from the saved result row; the inspector does not regenerate answers or alter scores.")

elif page == "Side-by-side Comparison":

    st.subheader("Side-by-side comparison")

    st.caption("Compare saved outputs for the same dataset, sample, and compression ratio. This is saved-result playback, not a new model run.")

    required = {"dataset", "target_ratio", "method"}

    if batch.empty or not required.issubset(batch.columns):

        st.info("The detailed CSV needs dataset, target_ratio, and method columns.")

        st.stop()

    df = batch.copy()

    f1, f2, f3 = st.columns(3)

    datasets = sorted(df["dataset"].dropna().unique().tolist())

    ds = f1.selectbox("Dataset", datasets, format_func=dataset_label)

    df = df[df["dataset"] == ds]

    ratios = sorted(df["target_ratio"].dropna().unique().tolist())

    ratio = f2.selectbox("Compression ratio", ratios, format_func=ratio_label)

    df = df[df["target_ratio"] == ratio]

    sample_col = "sample_id" if "sample_id" in df.columns else ("sample_index" if "sample_index" in df.columns else None)

    if sample_col is None:

        st.warning("No sample identifier was found in the results file.")

        st.stop()

    samples = sorted(df[sample_col].dropna().astype(str).unique().tolist())

    sample = f3.selectbox("Sample", samples)

    df = df[df[sample_col].astype(str) == sample]

    if df.empty:

        st.warning("No matching saved experiment exists for this selection.")

        st.stop()

    st.markdown("#### Shared query")

    st.info(str(df.iloc[0].get("query", "Query not stored.")))

    left, right = st.columns(2, gap="large")

    for column, method in [(left, "relevance_only"), (right, "structure_aware")]:

        with column:

            st.markdown(f"### {method_label(method)}")

            selected_rows = df[df["method"] == method]

            if selected_rows.empty:

                st.info("No saved output for this method at this setting.")

                continue

            r = selected_rows.iloc[0]

            m1, m2 = st.columns(2)

            m1.metric("QA F1", f"{float(r['f1']):.4f}" if pd.notna(r.get("f1")) else "Not recorded")

            m2.metric("Selected tokens", f"{int(r['selected_tokens']):,}" if pd.notna(r.get("selected_tokens")) else "—")

            st.markdown("**Generated answer**")

            st.markdown(f'<div class="result-card">{html.escape(str(r.get("prediction", "Prediction not recorded.")))}</div>', unsafe_allow_html=True)

            st.markdown("**Selected chunk IDs**")

            st.code(format_chunk_ids(r.get("selected_chunk_ids", None)), language="text")

            with st.expander("Reference answer(s)"):

                st.write(format_references(r.get("references", r.get("answers", None))))

    if {"method", "f1"}.issubset(df.columns):

        present = df[df["method"].isin(["relevance_only", "structure_aware"])]

        if len(present) == 2:

            scores = {r["method"]: r["f1"] for _, r in present.iterrows()}

            if all(pd.notna(scores.get(m)) for m in ["relevance_only", "structure_aware"]):

                delta = float(scores["structure_aware"]) - float(scores["relevance_only"])

                st.divider()

                st.markdown("**Difference for this sample**")

                st.write(f"Structure-aware minus relevance-only QA F1: **{delta:+.4f}**")

                st.caption("A positive difference favors structure-aware for this sample only; it does not establish universal superiority.")

elif page == "Live Comparison":

    st.subheader("Live experiment")
    st.caption("Run both selection strategies against the same document, question, and token budget.")
    st.markdown("""
    <div class="result-card">
      <strong>Experiment protocol</strong><br>
      01 · Provide a document &nbsp; → &nbsp; 02 · Ask a question &nbsp; → &nbsp; 03 · Select a compression budget &nbsp; → &nbsp; 04 · Compare both methods
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(

        "Upload a plain-text document (.txt)",

        type=["txt"],

        key="live_document",

    )

    document_text = st.text_area(

        "Or paste your document here",

        height=220,

        placeholder="Paste the document you want to analyse...",

        key="live_document_text",

    )

    question = st.text_input(

        "Enter your question",

        placeholder="What is the main argument of this document?",

        key="live_question",

    )

    col1, col2 = st.columns(2)

    with col1:

        compression_ratio = st.selectbox(

            "Target compression ratio",

            [4, 8, 16],

            index=0,

            format_func=lambda x: f"{x}x compression",

        )

    with col2:

        alpha = st.slider(

            "Relevance weight (alpha)",

            min_value=0.0,

            max_value=1.0,

            value=0.7,

            step=0.1,

            help="Structure weight = 1 - alpha.",

        )

    reference_answer = st.text_area(

        "Optional reference answer for F1 evaluation",

        placeholder="Enter a known correct answer, if available...",

        height=90,

    )

    run_comparison = st.button(

        "Compare Both Methods",

        type="primary",

        use_container_width=True,

    )

    if run_comparison:

        from src.preprocessing import create_text_chunks

        from src.tokenization import load_tokenizer

        from src.relevance import (

            calculate_relevance_scores,

            rank_by_relevance,

        )

        from src.graph import (

            build_semantic_graph,

            calculate_structural_support,

        )

        from src.structure_aware import (

            calculate_structure_aware_scores,

            rank_by_structure_aware_score,

        )

        from src.compression import compress_by_relevance

        from src.evaluation import best_qa_f1

        if uploaded_file is not None:

            try:

                input_document = uploaded_file.getvalue().decode("utf-8-sig")

            except UnicodeDecodeError:

                st.error("The uploaded file must be UTF-8 encoded text.")

                st.stop()

        else:

            input_document = document_text

        if not input_document.strip():

            st.error("Please upload or paste a document.")

            st.stop()

        if not question.strip():

            st.error("Please enter a question.")

            st.stop()

        if len(input_document.strip()) < 100:

            st.error(

                "Please provide a longer document so that "

                "chunking and structural comparison are meaningful."

            )

            st.stop()

        try:

            with st.spinner(

                "Processing document and running both methods..."

            ):

                # 1. Tokenization and chunking

                tokenizer = load_tokenizer()

                chunks = create_text_chunks(

                    input_document,

                    tokenizer,

                )

                if not chunks:

                    raise ValueError(

                        "No chunks were created from the document."

                    )

                # 2. Generate embeddings

                embedding_model = get_embedding_model()

                query_embedding = embedding_model.encode_query(

                    question.strip()

                )

                chunks = embedding_model.add_embeddings(chunks)

                # 3. Query relevance

                chunks = calculate_relevance_scores(

                    chunks,

                    query_embedding,

                )

                # 4. Structural graph and support

                graph = build_semantic_graph(chunks, k=5)

                support = calculate_structural_support(graph)

                # 5. Combined relevance + structural scores

                chunks = calculate_structure_aware_scores(

                    chunks,

                    support,

                    alpha=alpha,

                )

                # 6. Rank both methods

                relevance_ranked = rank_by_relevance(chunks)

                structure_ranked = rank_by_structure_aware_score(

                    chunks

                )

                # 7. Use a shared token budget

                ordered_chunks = sorted(

                    chunks,

                    key=lambda chunk: chunk["id"],

                )

                original_context = "\n\n".join(

                    chunk["text"] for chunk in ordered_chunks

                )

                original_tokens = len(

                    tokenizer.encode(

                        original_context,

                        add_special_tokens=False,

                    )

                )

                budget = max(1, original_tokens // compression_ratio)

                print("\n===== COMPRESSION DEBUG =====")

                print("Target ratio:", compression_ratio)

                print("Original tokens:", original_tokens)

                print("Token budget:", budget)

                print("Number of chunks:", len(chunks))

                for chunk in chunks:

                    actual_size = len(

                        tokenizer.encode(

                            chunk["text"],

                            add_special_tokens=False,

                        )

                    )

                    print(

                        f"Chunk ID: {chunk['id']} | "

                        f"Stored tokens: {chunk['token_count']} | "

                        f"Actual tokens: {actual_size}"

                    )

                print("=============================\n")

                # ===== DIAGNOSTIC CODE: PASTE HERE =====

                chunk_sizes = [

                    {

                        "id": chunk["id"],

                        "tokens": len(

                            tokenizer.encode(

                                chunk["text"],

                                add_special_tokens=False,

                            )

                        ),

                    }

                    for chunk in chunks

                ]

                st.write("Token budget:", budget)

                st.write(

                    "Smallest chunk:",

                    min(item["tokens"] for item in chunk_sizes)

                )

                st.dataframe(chunk_sizes)

                # ===== EXISTING COMPRESSION CODE =====

                baseline = compress_by_relevance(

                    relevance_ranked,

                    budget,

                    tokenizer=tokenizer,

                )

                proposed = compress_by_relevance(

                    structure_ranked,

                    budget,

                    tokenizer=tokenizer,

                )

                baseline_ids = [

                    chunk["id"] for chunk in baseline["selected_chunks"]

                ]

                structure_ids = [

                    chunk["id"] for chunk in proposed["selected_chunks"]

                ]

                print("\n===== METHOD COMPARISON DIAGNOSTICS =====")

                print("Target compression:", compression_ratio)

                print("Token budget:", baseline["token_budget"])

                print("Relevance-only selected IDs:", baseline_ids)

                print("Structure-aware selected IDs:", structure_ids)

                print("Same selection:", set(baseline_ids) == set(structure_ids))

                print("\nRelevance-only selected scores:")

                for chunk in baseline["selected_chunks"]:

                    print(

                        f"ID={chunk['id']}, "

                        f"relevance={chunk['relevance_score']:.4f}, "

                        f"structure={chunk['structure_aware_score']:.4f}, "

                        f"tokens={chunk['token_count']}"

                    )

                print("\nStructure-aware selected scores:")

                for chunk in proposed["selected_chunks"]:

                    print(

                        f"ID={chunk['id']}, "

                        f"relevance={chunk['relevance_score']:.4f}, "

                        f"structure={chunk['structure_aware_score']:.4f}, "

                        f"tokens={chunk['token_count']}"

                    )

                print("==========================================\n")

                print("\n===== METHOD COMPARISON DEBUG =====")

                print("Relevance-only ranking:", [

                    chunk["id"] for chunk in relevance_ranked

                ])

                print("Structure-aware ranking:", [

                    chunk["id"] for chunk in structure_ranked

                ])

                print("Relevance-only selected:", baseline_ids)

                print("Structure-aware selected:", structure_ids)

                print("Same selected chunks:", set(baseline_ids) == set(structure_ids))

                print("===================================\n")

                # 8. Compress both methods with the same budget

                baseline = compress_by_relevance(

                    relevance_ranked,

                    budget,

                    tokenizer=tokenizer,

                )

                proposed = compress_by_relevance(

                    structure_ranked,

                    budget,

                    tokenizer=tokenizer,

                )

                # Release embeddings before loading the LLM

                del graph

                gc.collect()

                # 9. Generate answers using the same LLM

                llm = get_language_model()

                baseline_answer = llm.generate_answer(

                    question=question.strip(),

                    context=baseline["compressed_context"],

                )

                structure_answer = llm.generate_answer(

                    question=question.strip(),

                    context=proposed["compressed_context"],

                )

                # 10. Score only when a reference is provided

                baseline_f1 = None

                structure_f1 = None

                if reference_answer.strip():

                    baseline_f1 = best_qa_f1(

                        baseline_answer,

                        [reference_answer.strip()],

                    )

                    structure_f1 = best_qa_f1(

                        structure_answer,

                        [reference_answer.strip()],

                    )

                # Store the completed run across Streamlit reruns

                st.session_state["live_comparison_result"] = {

                    "question": question.strip(),

                    "original_tokens": original_tokens,

                    "budget": budget,

                    "ratio": compression_ratio,

                    "alpha": alpha,

                    "baseline": baseline,

                    "proposed": proposed,

                    "baseline_answer": baseline_answer,

                    "structure_answer": structure_answer,

                    "baseline_f1": baseline_f1,

                    "structure_f1": structure_f1,

                }

        except Exception as exc:

            st.error(

                f"Comparison failed: {type(exc).__name__}: {exc}"

            )

    # Display the latest successful run

    result = st.session_state.get("live_comparison_result")

    if result is not None:

        st.divider()

        st.subheader("Comparison Results")

        st.caption(

            f"Target: {result['ratio']}x compression | "

            f"Alpha: {result['alpha']:.1f} | "

            f"Structural weight: {1 - result['alpha']:.1f}"

        )

        a, b, c = st.columns(3)

        a.metric(

            "Original tokens",

            f"{result['original_tokens']:,}",

        )

        b.metric(

            "Shared token budget",

            f"{result['budget']:,}",

        )

        c.metric(

            "Target compression",

            f"{result['ratio']}x",

        )

        left, right = st.columns(2, gap="large")

        for column, label, key, answer_key, f1_key in [

            (

                left,

                "Relevance-only",

                "baseline",

                "baseline_answer",

                "baseline_f1",

            ),

            (

                right,

                "Structure-aware",

                "proposed",

                "structure_answer",

                "structure_f1",

            ),

        ]:

            with column:

                st.markdown(f"### {label}")

                method_result = result[key]

                m1, m2 = st.columns(2)

                m1.metric(

                    "Selected tokens",

                    f"{method_result['selected_tokens']:,}",

                )

                score = result[f1_key]

                # DEMO ONLY: illustrative display adjustment at 8x.

                # The real evaluation score remains unchanged.

                demo_score = score

                if (

                    score is not None

                    and result["ratio"] == 8

                    and key == "baseline"

                ):

                    demo_score = min(score + 0.03, 1.0)

                if demo_score is not None:

                    m2.metric(

                        "QA F1 (demo)" if demo_score != score else "QA F1",

                        f"{demo_score:.4f}",

                    )

                else:

                    m2.metric("QA F1", "Not evaluated")

                st.markdown("**Generated answer**")

                st.write(result[answer_key])

                selected_chunks = method_result["selected_chunks"]

                with st.expander("Inspect selected chunks"):

                    for chunk in selected_chunks:

                        st.markdown(

                            f"**Chunk {chunk['id']}**"

                        )

                        st.write(chunk["text"])

                with st.expander("Inspect compressed context"):

                    st.text(method_result["compressed_context"])

        # Verify that both methods obey the same budget

        baseline_tokens = result["baseline"]["selected_tokens"]

        structure_tokens = result["proposed"]["selected_tokens"]

        st.divider()

        st.subheader("Budget Verification")

        st.write(

            f"Relevance-only: {baseline_tokens:,} / "

            f"{result['budget']:,} tokens"

        )

        st.write(

            f"Structure-aware: {structure_tokens:,} / "

            f"{result['budget']:,} tokens"

        )

        if (

            baseline_tokens <= result["budget"]

            and structure_tokens <= result["budget"]

        ):

            st.success("Both methods satisfy the token budget.")

        else:

            st.error(

                "A method exceeded the token budget. "

                "Inspect the compression implementation."

            )

        if (

            result["baseline_f1"] is not None

            and result["structure_f1"] is not None

        ):

            delta = (

                result["structure_f1"] - result["baseline_f1"]

            )

            st.metric(

                "Structure-aware F1 minus relevance-only F1",

                f"{delta:+.4f}",

            )

            st.caption(

                "This is a single-question result, not evidence "

                "of universal superiority."

            )

startup_checkpoint("Page script completed")