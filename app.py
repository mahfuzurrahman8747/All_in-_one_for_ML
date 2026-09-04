"""
Auto‑ML Streamlit App
=====================

This lightweight app lets a user drag‑and‑drop a CSV or Excel file, automatically
detects whether the task is a classification or regression problem, trains a simple
baseline model (RandomForest), evaluates it, and visualises the result.

The implementation follows the plan approved earlier – it is deliberately short
and heavily commented so the logic is clear.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 1️⃣  Setup – ensure an uploads folder exists.
# ---------------------------------------------------------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2️⃣  UI – title and file uploader.
# ---------------------------------------------------------------------------
st.title("📊 Auto‑ML – Upload a dataset and get instant results")
st.write(
    "Upload a CSV or Excel file. The app will infer the problem type, "
    "train a baseline model, and show the best‑fit metric and a plot."
)

uploaded_file = st.file_uploader(
    "🗂️ Drag‑and‑drop your data file here",
    type=["csv", "xlsx"],
    help="Supported formats: CSV and Excel (.xlsx)",
)

if uploaded_file is not None:
    # -------------------------------------------------------------------
    # 3️⃣  Persist the uploaded file to the backend.
    # -------------------------------------------------------------------
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved to `{file_path}`")

    # -------------------------------------------------------------------
    # 4️⃣  Load the data into a pandas DataFrame.
    # -------------------------------------------------------------------
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            # pandas uses openpyxl for .xlsx files – it is a lightweight
            # dependency that is usually present in a standard Python env.
            df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"❌ Failed to read the uploaded file: {e}")
        st.stop()

    st.subheader("Dataset preview (first 5 rows)")
    st.dataframe(df.head())

    # -------------------------------------------------------------------
    # 5️⃣  Let the user pick the target/label column.
    # -------------------------------------------------------------------
    target_col = st.selectbox("🎯 Choose the target column", options=df.columns)
    y = df[target_col]

else:
    st.info("📁 Awaiting a data file upload…")
