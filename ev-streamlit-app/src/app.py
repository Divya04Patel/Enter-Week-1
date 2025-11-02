import streamlit as st
from components.sidebar import sidebar_navigation
from pages.data_cleaning import data_cleaning_page
from pages.visualization import visualization_page

# Load custom CSS
st.markdown('<link href="styles/styles.css" rel="stylesheet">', unsafe_allow_html=True)

# Set up sidebar for navigation
sidebar_navigation()

# Main application logic
st.title("EV Data Analysis App")

# Display the selected page
page = st.sidebar.selectbox("Select a page:", ["Data Cleaning", "Visualization"])

if page == "Data Cleaning":
    data_cleaning_page()
elif page == "Visualization":
    visualization_page()