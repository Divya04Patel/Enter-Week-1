import streamlit as st

def sidebar():
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select a page:", ("Data Cleaning", "Visualization"))

    if options == "Data Cleaning":
        return "data_cleaning"
    elif options == "Visualization":
        return "visualization"