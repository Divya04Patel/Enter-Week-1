# webapp/app.py
import streamlit as st
import pandas as pd
import io
import sys
import os
from pathlib import Path

# Make sure Streamlit can find your preprocessing script (resolve relative to this file)
base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "src"
sys.path.append(str(src_dir))

# Add components
sys.path.append(str(base_dir / "webapp"))  # ensure components package is importable
from components.predict_page import predict_page  # new
try:
    from data_preprocessing import clean_data
    CLEAN_IMPORT_ERROR = None
except Exception as e:
    clean_data = None
    CLEAN_IMPORT_ERROR = e

st.set_page_config(page_title="EV Data Cleaning App", layout="wide")
st.sidebar.title("⚡ EV Dashboard")
page = st.sidebar.radio("Navigate", ["Data Cleaning", "Visualization", "Predict"])

# call pages
if page == "Predict":
    predict_page()
elif page == "Visualization":
    # keep your existing visualization import/call (if present)
    try:
        from components.visualization_page import visualization_page
        visualization_page()
    except Exception:
        st.warning("Visualization page not available")
else:
    # Data Cleaning UI (the rest of your existing file remains unchanged and runs here)
    st.title("⚡ EV Dataset Cleaning Dashboard")
    st.write("Upload your dataset, preview it, and clean it automatically using smart preprocessing.")

    # Sample dataset helper
    sample_path = base_dir / "data" / "ev_dataset.csv"

    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
    if st.button("Use sample dataset") and sample_path.exists():
        # open sample as binary file-like for Streamlit
        uploaded_file = open(sample_path, "rb")

    if CLEAN_IMPORT_ERROR:
        st.error(f"Error importing preprocessing module: {CLEAN_IMPORT_ERROR}")
        st.stop()

    # cache wrapper around external clean_data
    @st.cache_data
    def cached_clean(df: pd.DataFrame):
        # ensure we pass a proper DataFrame to clean_data
        return clean_data(df)

    if uploaded_file is not None:
        # Read uploaded data
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        st.subheader("📊 Raw Data Preview")
        left, right = st.columns([2, 1])

        with left:
            st.dataframe(df.head(10))
            st.write(f"*Shape:* {df.shape[0]} rows × {df.shape[1]} columns")
            with st.expander("Show column types and missing counts"):
                st.write(df.dtypes.astype(str).to_frame("dtype"))
                st.write("Missing values:")
                st.dataframe(df.isnull().sum().reset_index().rename(columns={'index':'Column',0:'Missing Count'}))

        with right:
            st.markdown("### Actions")
            st.write("Click to clean data using project preprocessing.")
            initial_rows = df.shape[0]
            dup_before = df.duplicated().sum()
            if st.button("🧹 Clean Data"):
                with st.spinner("Cleaning dataset..."):
                    cleaned_df = cached_clean(df)
                rows_after = cleaned_df.shape[0]
                dup_after = cleaned_df.duplicated().sum()
                rows_removed = initial_rows - rows_after
                dup_removed = int(dup_before - dup_after)

                st.success("Cleaning finished")
                st.metric("Rows (before → after)", f"{initial_rows} → {rows_after}", delta=f"-{rows_removed}")
                st.metric("Duplicates removed", dup_removed)

                # show cleaned preview and basic plots
                st.subheader("✅ Cleaned Data Preview")
                st.dataframe(cleaned_df.head(10))

                with st.expander("Summary statistics"):
                    st.dataframe(cleaned_df.describe(include='all'))

                # Basic charts (guard on available columns)
                chart_cols = []
                for c in ['Range_km', 'Battery_Capacity_kWh', 'Power_hp']:
                    if c in cleaned_df.columns:
                        chart_cols.append(c)

                if chart_cols:
                    st.subheader("Quick feature charts")
                    for col in chart_cols:
                        st.write(f"Distribution: {col}")
                        st.bar_chart(cleaned_df[col].dropna().value_counts().sort_index())

                # Download cleaned CSV
                buffer = io.BytesIO()
                cleaned_df.to_csv(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="💾 Download Cleaned CSV",
                    data=buffer,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv"
                )

    else:
        st.info("👆 Upload a CSV file or use the sample dataset to begin.")