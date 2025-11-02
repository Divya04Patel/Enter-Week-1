# webapp/components/visualization_page.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualization_page():
    st.title("📈 Data Visualization")
    uploaded_file = st.file_uploader("Upload CSV for visualization", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.write("### Column-wise Distribution")
        col = st.selectbox("Select a column to visualize", df.columns)
        fig2, ax2 = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax2)
        st.pyplot(fig2)
    else:
        st.info("👆 Upload a CSV file to visualize data.")