import streamlit as st
import pandas as pd
import io
import sys
import os

sys.path.append(os.path.abspath("../src"))
from data_preprocessing import clean_data

def data_cleaning_page():
    st.set_page_config(page_title="EV Data Cleaning", layout="wide")
    st.title("⚡ EV Dataset Cleaning Dashboard")
    st.write("Upload your dataset, preview it, and clean it automatically using smart preprocessing.")

    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head(10))

        st.write(f"*Shape:* {df.shape[0]} rows × {df.shape[1]} columns")
        st.write("*Missing Values:*")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={'index':'Column',0:'Missing Count'}))

        if st.button("🧹 Clean Data"):
            cleaned_df = clean_data(df)
            
            st.subheader("✅ Cleaned Data Preview")
            st.dataframe(cleaned_df.head(10))

            st.write("*Shape after cleaning:*", cleaned_df.shape)
            st.write("*Summary statistics:*")
            st.dataframe(cleaned_df.describe(include='all'))

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
        st.info("👆 Upload a CSV file to begin.")