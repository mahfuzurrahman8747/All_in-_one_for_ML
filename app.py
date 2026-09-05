import json
from pathlib import Path

import pandas as pd
import streamlit as st
from model_train import train_and_save_model


UPLOAD_DIR = Path("uploads")
STATE_FILE = Path("selected_data.json")
RESULTS_FILE = Path("best_model_results.json")
MODEL_FILE = Path("model_parameters/best_model.joblib")
UPLOAD_DIR.mkdir(exist_ok=True)

def setup_page() -> None:
    """Configure the page and keep the visual styling in one place."""
    st.set_page_config(page_title="Model results", page_icon=":material/insights:", layout="centered")
    st.markdown(
        """
        <style>
            .stApp { background: #f5f3ef; color: #1e2927; }
            .stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"],
            .stApp [data-testid="stFileUploaderDropzoneInstructions"],
            .stApp [data-testid="stFileUploaderDropzoneInstructions"] span,
            .stApp [data-testid="stMetricValue"] { color: #1e2927 !important; }
            .stApp [data-testid="stFileUploaderDropzone"] { background: #fffdf9; border: 1px dashed #c9c0b4; }
            .stApp [data-testid="stFileUploaderDropzone"] small,
            .stApp [data-testid="stMetricLabel"] { color: #61706c !important; }
            .block-container { max-width: 860px; padding-top: 4rem; padding-bottom: 4rem; }
            .eyebrow, .result-label { color: #b4553d; font-size: .75rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
            .hero-title { color: #1e2927; font-size: 4.2rem; font-weight: 700; letter-spacing: -.04em; line-height: .98; margin: .35rem 0 .9rem; }
            .hero-copy { color: #61706c; font-size: 1.05rem; margin-bottom: 2.2rem; }
            .model-name { color: #1e2927; font-size: 1.8rem; font-weight: 700; margin: .2rem 0 .25rem; }
            div[data-testid="stMetric"] { background: #fffdf9; border: 1px solid #e5dfd5; border-radius: 12px; padding: 1rem 1.1rem; }
            @media (max-width: 640px) { .hero-title { font-size: 2.7rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_header() -> None:
    st.markdown('<div class="eyebrow">Auto ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Your model,<br>clearly measured.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Upload data, choose a target, and review the latest saved model result.</div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def load_results(file_path: str, modified_time: float) -> dict:
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


def upload_and_train() -> None:
    uploaded_file = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx"],
        help="Supported formats: CSV and Excel (.xlsx)",
    )

    if uploaded_file is not None:
        file_path = UPLOAD_DIR / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())

        try:
            df = pd.read_csv(file_path) if file_path.suffix.lower() == ".csv" else pd.read_excel(file_path)
        except Exception as error:
            st.error(f"Could not read this file: {error}")
            return

        target_col = st.selectbox(
            "Choose the target column",
            options=df.columns,
            index=None,
            placeholder="Select a target column",
        )

        if target_col is None:
            st.info("Select a target column to continue.")
            return

        STATE_FILE.write_text(
            json.dumps({"uploaded_file_name": uploaded_file.name, "target_col": target_col}, indent=2),
            encoding="utf-8",
        )

        if st.button("Train model", type="primary"):
            try:
                with st.spinner("Training model..."):
                    train_and_save_model(data_path=file_path, target_col=target_col)
                st.success("Model trained successfully.")
                st.rerun()
            except ValueError as error:
                st.error(f"Training could not start: {error}")
            except Exception as error:
                st.error(f"Training failed: {error}")


def show_results() -> None:
    if not RESULTS_FILE.exists():
        st.caption("Run the notebook to create the first saved model result.")
        return

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
            st.metric("Accuracy", f"{metrics.get('Accuracy', 0):.2f}", border=True)
        else:
            metric_columns = st.columns(4)
            with metric_columns[0]:
                st.metric("Accuracy", f"{metrics.get('Accuracy', 0):.2f}", border=True)
            with metric_columns[1]:
                st.metric("MAE", f"{metrics.get('MAE', 0):.2f}", border=True)
            with metric_columns[2]:
                st.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}", border=True)
            with metric_columns[3]:
                st.metric("R²", f"{metrics.get('R2', 0):.4f}", border=True)

        if MODEL_FILE.is_file():
            st.download_button(
                "Download best model",
                data=MODEL_FILE.read_bytes(),
                file_name="best_model.joblib",
                mime="application/octet-stream",
                type="primary",
                width="stretch",
            )

    with st.expander("How to use the downloaded model"):
        st.markdown(
            """
            Download `best_model.joblib` after training. The file contains the trained model,
            scaler, transformed feature-column names, target column, and problem type.

            Install the required packages, then load the bundle like this:

            ```python
            import joblib

            bundle = joblib.load("best_model.joblib")
            model = bundle["model"]
            scaler = bundle["scaler"]
            feature_columns = bundle["feature_columns"]
            target_classes = bundle.get("target_classes")

            # Prepare new_data with the same cleaning and one-hot encoding used during training.
            new_data = new_data.reindex(columns=feature_columns, fill_value=0)
            scaled_data = scaler.transform(new_data)
            predictions = model.predict(scaled_data)

            if target_classes is not None:
                predictions = [target_classes[int(value)] for value in predictions]
            print(predictions)
            ```

            New data must use the same input columns and preprocessing as the training data.
            Keep the downloaded file with the same scikit-learn and model-library versions
            used for training.
            """
        )


setup_page()
show_header()
with st.container(border=True):
    upload_and_train()
show_results()