from pathlib import Path
import ast
import json
import math

import pandas as pd
import plotly.express as px
import streamlit as st

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
.block-container {padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1480px;}
[data-testid="stSidebar"] {border-right: 1px solid #e3e8f0;}
.hero {
  padding: 1.35rem 1.55rem; border-radius: 18px;
  background: linear-gradient(115deg, #edf2ff 0%, #edf8f6 100%);
  border: 1px solid #dce5f1; margin-bottom: 1.15rem; color: #182943;
}
.hero h1 {font-size: 2.05rem; line-height: 1.2; margin: 0 0 .45rem 0;}
.hero p {margin: 0; color: #52647b; font-size: 1rem;}
.eyebrow {font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; color: #526b94; font-weight: 700; margin-bottom: .4rem;}
.subtle {color: #6b778b; font-size: .88rem;}
div[data-testid="stMetric"] {background: #f7f9fc; border: 1px solid #e4eaf3; padding: .7rem .85rem; border-radius: 13px;}
div[data-testid="stMetricLabel"] {font-size: .82rem;}
.result-card {border: 1px solid #e1e7f0; border-radius: 13px; padding: 1rem; background: #fbfcfe;}
hr {border-color: #e7ebf2;}
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


@st.cache_data(ttl=20)
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


batch, summary = load_results()

with st.sidebar:
    st.markdown("## 🧠 Context Compression Lab")
    st.caption("Research evaluation workspace")
    page = st.radio(
        "Navigate",
        ["Overview", "Results Explorer", "Experiment Inspector", "Side-by-side Comparison"],
        label_visibility="visible",
    )
    st.divider()
    if not batch.empty:
        st.caption(f"**{len(batch)}** saved experiment rows")
        st.caption("Data source: `results/*.csv`")
    else:
        st.caption("Results files not found")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Research evaluation workspace</div>
  <h1>Context Compression Lab</h1>
  <p>Evaluating Structural Information for Query-Aware Context Compression in Long-Context LLMs</p>
</div>
""", unsafe_allow_html=True)

if batch.empty and summary.empty:
    st.warning("No results found. Place `batch_results.csv` and `summary.csv` inside the project's `results` folder.")
    st.code("Context_Compression/\n├── app.py\n├── src/\n└── results/\n    ├── batch_results.csv\n    └── summary.csv")
    st.stop()

if "dataset" not in batch.columns and not batch.empty:
    st.error("The detailed CSV needs a `dataset` column.")
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
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Higher QA F1 means greater token overlap with the reference answer. The displayed sample count is essential when interpreting averages.")

        display = chart_df.copy()
        display["Dataset"] = display["dataset"].map(dataset_label)
        keep = [c for c in ["Dataset", "Compression", "Method", "sample_count", "mean_f1", "mean_f1_percent"] if c in display.columns]
        st.dataframe(
            display[keep].rename(columns={"sample_count": "Samples", "mean_f1": "Mean F1", "mean_f1_percent": "Mean F1 (%)"}),
            use_container_width=True, hide_index=True,
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
        st.latex(r"S_i = \alpha R_i + (1-\alpha)G_i")
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
        st.markdown(f'<div class="result-card">{format_references(row.get("references", row.get("answers", None)))}</div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Generated answer")
        st.markdown(f'<div class="result-card">{str(row.get("prediction", "Prediction not stored."))}</div>', unsafe_allow_html=True)

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
            st.markdown(f'<div class="result-card">{str(r.get("prediction", "Prediction not recorded."))}</div>', unsafe_allow_html=True)
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
