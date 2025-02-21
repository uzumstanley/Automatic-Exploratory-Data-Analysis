import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from eda_tool.data_loader import load_data
from eda_tool.eda_summary import dataset_summary
from eda_tool.missing_values import plot_missing_values
from eda_tool.visualization import (
    plot_histograms, plot_correlation_matrix, plot_countplots, 
    plot_boxplots, plot_violinplots, plot_pairplot, plot_kde
)

st.title("Automatic EDA Tool")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("Click Below to Explore the Dataset")

        with st.expander("Show Dataset Shape"):
            st.write(f"Shape: {df.shape}")

        with st.expander("Show Column Names"):
            st.write(f"Columns: {list(df.columns)}")

        with st.expander("Show Missing Values"):
            st.write(df.isnull().sum())

        with st.expander("Show Data Types"):
            st.write(df.dtypes)

        with st.expander("Show Summary Statistics"):
            st.write(df.describe())

        with st.expander("Show Missing Values Heatmap"):
            st.pyplot(plot_missing_values(df))

        with st.expander("Show Feature Distributions (Histogram)"):
            st.pyplot(plot_histograms(df))

        with st.expander("Show Correlation Matrix"):
            st.pyplot(plot_correlation_matrix(df))

        with st.expander("Show Count Plots for Categorical Features"):
            for fig in plot_countplots(df):
                st.pyplot(fig)

        with st.expander("Show Box Plots for Outlier Detection"):
            for fig in plot_boxplots(df):
                st.pyplot(fig)

        with st.expander("Show Violin Plots"):
            for fig in plot_violinplots(df):
                st.pyplot(fig)

        with st.expander("Show Pair Plot (Feature Relationships)"):
            st.pyplot(plot_pairplot(df))

        with st.expander("Show KDE (Density) Plots"):
            for fig in plot_kde(df):
                st.pyplot(fig)

        # PCA (Principal Component Analysis)
        with st.expander("Show PCA (Principal Component Analysis)"):
            numeric_df = df.select_dtypes(include=["number"])
            if not numeric_df.empty:
                pca = PCA(n_components=min(3, len(numeric_df.columns)))  # Max 3 components
                pca_result = pca.fit_transform(numeric_df)
                explained_variance = pca.explained_variance_ratio_

                st.write(f"Explained Variance Ratio: {explained_variance}")
                fig, ax = plt.subplots()
                ax.bar(range(len(explained_variance)), explained_variance)
                ax.set_xlabel("Principal Components")
                ax.set_ylabel("Explained Variance")
                ax.set_title("PCA Explained Variance")
                st.pyplot(fig)
            else:
                st.write("No numeric columns available for PCA.")

        # Feature Importance (Random Forest)
        with st.expander("Show Feature Importance (Random Forest)"):
            df_clean = df.dropna()  # Drop missing values
            target_column = st.selectbox("Select Target Column for Feature Importance", df.columns)
            
            if target_column:
                X = df_clean.drop(columns=[target_column])
                y = df_clean[target_column]

                # Convert categorical columns to numerical
                for col in X.select_dtypes(include=['object']).columns:
                    X[col] = LabelEncoder().fit_transform(X[col])

                # Check if target is categorical or numerical
                if y.dtype == 'O' or len(y.unique()) < 10:  
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    y = LabelEncoder().fit_transform(y)  # Encode target if categorical
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)

                model.fit(X, y)
                importance = model.feature_importances_

                feature_importance_df = (
                    pd.DataFrame({'Feature': X.columns, 'Importance': importance})
                    .sort_values(by="Importance", ascending=False)
                )

                st.write(feature_importance_df)
                fig, ax = plt.subplots()
                sns.barplot(data=feature_importance_df, x="Importance", y="Feature", ax=ax)
                ax.set_title("Feature Importance (Random Forest)")
                st.pyplot(fig)

    else:
        st.error("Error while loading the dataset.")
