import json
from pathlib import Path

import pandas as pd
import streamlit as st
from model_train import train_and_save_model


UPLOAD_DIR = Path("uploads")
STATE_FILE = Path("selected_data.json")
RESULTS_FILE = Path("best_model_results.json")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Model results",
    page_icon=":material/insights:",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #f5f3ef;
            color: #1e2927;
        }
        .stApp p,
        .stApp label,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stFileUploaderDropzoneInstructions"],
        .stApp [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: #1e2927 !important;
        }
        .stApp [data-testid="stFileUploaderDropzone"] {
            background: #fffdf9;
            border: 1px dashed #c9c0b4;
        }
        .stApp [data-testid="stFileUploaderDropzone"] small {
            color: #61706c !important;
        }
        .stApp [data-testid="stMetricLabel"] {
            color: #61706c !important;
        }
        .stApp [data-testid="stMetricValue"] {
            color: #1e2927 !important;
        }
        .block-container {
            max-width: 860px;
            padding-top: 4rem;
            padding-bottom: 4rem;
        }
        .eyebrow {
            color: #b4553d;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .hero-title {
            color: #1e2927;
            font-size: 4.2rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            line-height: 0.98;
            margin: 0.35rem 0 0.9rem;
        }
        .hero-copy {
            color: #61706c;
            font-size: 1.05rem;
            margin-bottom: 2.2rem;
        }
        .result-label {
            color: #61706c;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .model-name {
            color: #1e2927;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0.2rem 0 0.25rem;
        }
        div[data-testid="stMetric"] {
            background: #fffdf9;
            border: 1px solid #e5dfd5;
            border-radius: 12px;
            padding: 1rem 1.1rem;
        }
        @media (max-width: 640px) {
            .hero-title {
                font-size: 2.7rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Auto ML</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Your model,<br>clearly measured.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Upload data, choose a target, and review the latest saved model result.</div>',
    unsafe_allow_html=True,
)


@st.cache_data
def load_results(file_path: str, modified_time: float) -> dict:
    with open(file_path, encoding="utf-8") as results_file:
        return json.load(results_file)


with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx"],
        help="Supported formats: CSV and Excel (.xlsx)",
    )

    if uploaded_file is not None:
        file_path = UPLOAD_DIR / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())

        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception as error:
            st.error(f"Could not read this file: {error}")
            st.stop()

        target_col = st.selectbox("Choose the target column", options=df.columns)
        with STATE_FILE.open("w", encoding="utf-8") as state_file:
            json.dump(
                {
                    "uploaded_file_name": uploaded_file.name,
                    "target_col": target_col,
                },
                state_file,
                indent=2,
            )
            # Train the model using the uploaded data and selected target.
            train_and_save_model(data_path=file_path, target_col=target_col)


if RESULTS_FILE.exists():
    results = load_results(str(RESULTS_FILE), RESULTS_FILE.stat().st_mtime)
    metrics = results.get("metrics", {})
    problem_type = results.get("problem_type", "").title()

    st.space("large")
    with st.container(border=True):
        st.markdown('<div class="result-label">Latest saved result</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="model-name">{results.get("model_name", "Unknown model")}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"{problem_type} · target: {results.get('target_column', 'Unknown')}"
        )

        if results.get("problem_type") == "classification":
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}", border=True)
        else:
            metric_columns = st.columns(3)
            with metric_columns[0]:
                st.metric("R²", f"{metrics.get('r2', 0):.4f}", border=True)
            with metric_columns[1]:
                st.metric("MAE", f"{metrics.get('mae', 0):.2f}", border=True)
            with metric_columns[2]:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}", border=True)
else:
    st.caption("Run the notebook to create the first saved model result.")