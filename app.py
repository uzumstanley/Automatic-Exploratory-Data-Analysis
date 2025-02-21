import streamlit as st
import pandas as pd
from eda_tool.data_loader import load_data
from eda_tool.visualization import (
    plot_histograms, plot_correlation_matrix, perform_pca, feature_importance
)

# Streamlit app title
st.title("Automatic Exploratory Data Analysis Tool")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.write("### Preview of Data")
    st.write(df.head())

    # Data summary
    st.write("### Dataset Information")
    st.write(df.describe())

    # Histogram visualization
    with st.expander("Show Histograms"):
        st.pyplot(plot_histograms(df))

    # Correlation matrix
    with st.expander("Show Correlation Matrix"):
        correlation_fig = plot_correlation_matrix(df)
        if correlation_fig:
            st.pyplot(correlation_fig)
        else:
            st.write("No numeric columns available for correlation matrix.")

    # PCA
    with st.expander("Show PCA (Principal Component Analysis)"):
        explained_variance, pca_fig = perform_pca(df)
        if explained_variance is not None:
            st.write(f"Explained Variance Ratio: {explained_variance}")
            st.pyplot(pca_fig)
        else:
            st.write("No numeric columns available for PCA.")

    # Feature Importance
    with st.expander("Show Feature Importance (Random Forest)"):
        target_column = st.selectbox("Select Target Column for Feature Importance", df.columns)
        if target_column:
            feature_importance_df, feature_fig = feature_importance(df, target_column)
            st.write(feature_importance_df)
            st.pyplot(feature_fig)
