# webapp/components/data_cleaning_page.py
import streamlit as st
import pandas as pd
import io
import sys, os

# Import data cleaning function from src
sys.path.append(os.path.abspath("../src"))
from data_preprocessing import clean_data

def data_cleaning_page():
    st.title("📂 Data Upload & Cleaning")
    st.write("Upload your CSV dataset, preview it, and clean it automatically using preprocessing logic.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10))

        st.write(f"*Shape:* {df.shape[0]} rows × {df.shape[1]} columns")

        if st.button("🧹 Clean Data"):
            cleaned_df = clean_data(df)
            st.success("✅ Data cleaned successfully!")
            st.dataframe(cleaned_df.head(10))

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
        st.info("👆 Upload a CSV file to start.")