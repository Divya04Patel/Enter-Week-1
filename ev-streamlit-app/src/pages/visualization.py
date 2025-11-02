from streamlit import st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualization_page(cleaned_data):
    st.title("📊 Data Visualization Page")

    if cleaned_data is not None:
        st.subheader("Visualize Your Cleaned Data")

        # Select a column for visualization
        column = st.selectbox("Select a column to visualize", cleaned_data.columns)

        # Choose the type of plot
        plot_type = st.selectbox("Select plot type", ["Histogram", "Box Plot", "Scatter Plot"])

        if plot_type == "Histogram":
            st.subheader(f"Histogram of {column}")
            plt.figure(figsize=(10, 5))
            sns.histplot(cleaned_data[column], bins=30, kde=True)
            st.pyplot(plt)

        elif plot_type == "Box Plot":
            st.subheader(f"Box Plot of {column}")
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=cleaned_data[column])
            st.pyplot(plt)

        elif plot_type == "Scatter Plot":
            if len(cleaned_data.columns) < 2:
                st.warning("Need at least two columns for scatter plot.")
            else:
                x_column = st.selectbox("Select X-axis column", cleaned_data.columns)
                y_column = st.selectbox("Select Y-axis column", cleaned_data.columns)
                st.subheader(f"Scatter Plot of {x_column} vs {y_column}")
                plt.figure(figsize=(10, 5))
                sns.scatterplot(data=cleaned_data, x=x_column, y=y_column)
                st.pyplot(plt)

    else:
        st.info("Please clean your data first to visualize it.")