import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from evaluation import (MODEL_FILES, CLASS_NAMES, load_model, split_xy, evaluate, confusion,
                        report_text, roc_points)
from features import FEATURE_COLUMNS

st.set_page_config(page_title="Netflix Movie-vs-TV Classifier", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(90deg,#141414 0%,#b20710 100%); padding: 1.4rem 1.8rem;
           border-radius: 14px; color: #fff; margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2rem;} .hero p {margin: .3rem 0 0; opacity: .85;}
    </style>
    <div class="hero"><h1>🎬 Netflix Catalog: Movie or TV Show?</h1>
    <p>Five classical ML models predicting a title's type from its metadata · BITS Pilani M.Tech (DSE) · ML Assignment 2</p></div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("⚙️ Controls")
uploaded = st.sidebar.file_uploader("1 · Upload test data (CSV)", type=["csv"],
                                    help="Use test_data.csv from the repository (engineered features + `target`).")
use_sample = False
if os.path.exists("test_data.csv"):
    use_sample = st.sidebar.checkbox("…or use the bundled test_data.csv", value=False)
model_name = st.sidebar.selectbox("2 · Choose a model", list(MODEL_FILES.keys()))
threshold = st.sidebar.slider("3 · Decision threshold for 'TV Show'", 0.05, 0.95, 0.50, 0.05,
                              help="0.50 is the default used for the README table.")
st.sidebar.caption("Target: 1 = TV Show, 0 = Movie. `duration` is intentionally excluded (it would leak the label).")


@st.cache_resource(show_spinner=False)
def cached_model(name):
    return load_model(name)


@st.cache_data(show_spinner=False)
def read_csv(src):
    return pd.read_csv(src)


df = None
if uploaded is not None:
    df = read_csv(uploaded)
elif use_sample:
    df = read_csv("test_data.csv")

tab_eval, tab_compare, tab_about = st.tabs(["📊 Model evaluation", "🏆 Compare all models", "ℹ️ About the data"])

with tab_about:
    st.subheader("Expected CSV schema")
    st.write(f"{len(FEATURE_COLUMNS)} engineered feature columns + `target`. Column order is not important; extra columns are ignored.")
    st.code(", ".join(FEATURE_COLUMNS) + ", target", language="text")
    st.markdown(
        "- **Source:** Netflix Movies & TV Shows catalogue (8,807 titles, Kaggle).\n"
        "- **Task:** binary classification — Movie (0) vs TV Show (1).\n"
        "- **Split:** 80/20 stratified; `test_data.csv` is the 20 % hold-out (1,762 rows)."
    )
    if os.path.exists("model/metrics_summary.csv"):
        st.markdown("**Hold-out results recorded at training time**")
        st.dataframe(pd.read_csv("model/metrics_summary.csv"), hide_index=True, use_container_width=True)

if df is None:
    with tab_eval:
        st.info("👈 Upload `test_data.csv` (or tick the bundled-file box) in the sidebar to evaluate a model.")
    with tab_compare:
        st.info("Upload the test data first.")
    st.stop()

try:
    X, y = split_xy(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.sidebar.success(f"{len(df):,} rows · {y.mean():.1%} TV Shows")

# ---------------------------------------------------------------- single model
with tab_eval:
    model = cached_model(model_name)
    metrics, pred, proba = evaluate(model, X, y, threshold)
    st.subheader(f"{model_name} — hold-out performance")
    cols = st.columns(6)
    for c, (k, v) in zip(cols, metrics.items()):
        c.metric(k, f"{v:.4f}")

    left, mid, right = st.columns([1, 1, 1])
    with left:
        st.markdown("##### Confusion matrix")
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap(confusion(y, pred), annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with mid:
        st.markdown("##### ROC curve")
        fpr, tpr = roc_points(y, proba)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.plot(fpr, tpr, color="#b20710", lw=2, label=f"AUC = {metrics['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
        st.pyplot(fig)
    with right:
        st.markdown("##### Classification report")
        st.code(report_text(y, pred), language="text")

    with st.expander("Peek at predictions"):
        view = df.copy()
        view["P(TV Show)"] = proba.round(4)
        view["predicted"] = pred
        view["correct"] = view["predicted"] == y
        st.dataframe(view[["target", "predicted", "P(TV Show)", "correct"]].head(200), use_container_width=True)

# ---------------------------------------------------------------- comparison
with tab_compare:
    st.subheader("All five models on the uploaded data")
    rows = []
    for name in MODEL_FILES:
        m, _, _ = evaluate(cached_model(name), X, y, threshold)
        rows.append({"ML Model Name": name, **m})
    table = pd.DataFrame(rows)
    st.dataframe(table.style.format({c: "{:.4f}" for c in table.columns[1:]})
                 .highlight_max(subset=table.columns[1:], color="#f8d7da"),
                 hide_index=True, use_container_width=True)
    metric = st.radio("Plot metric", list(table.columns[1:]), horizontal=True, index=4)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(table["ML Model Name"], table[metric], color="#b20710")
    ax.set_xlim(max(0, table[metric].min() - 0.05), 1.0)
    for i, v in enumerate(table[metric]):
        ax.text(v, i, f" {v:.4f}", va="center")
    ax.invert_yaxis(); ax.set_xlabel(metric)
    st.pyplot(fig)
    best = table.loc[table["MCC"].idxmax(), "ML Model Name"]
    st.success(f"Highest MCC at threshold {threshold:.2f}: **{best}**")
