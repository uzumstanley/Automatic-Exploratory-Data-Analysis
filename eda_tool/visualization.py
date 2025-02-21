import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def plot_histograms(df):
    """Plots histograms for all numeric columns in the DataFrame."""
    df.hist(figsize=(10, 6), bins=30)
    plt.suptitle("Histograms of Numeric Features")
    plt.show()

def plot_correlation_matrix(df):
    """Plots the correlation matrix heatmap."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return None
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    return plt

def perform_pca(df):
    """Performs Principal Component Analysis (PCA) on numeric features and plots explained variance."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return None, None

    pca = PCA(n_components=min(3, len(numeric_df.columns)))
    pca_result = pca.fit_transform(numeric_df)
    explained_variance = pca.explained_variance_ratio_

    fig, ax = plt.subplots()
    ax.bar(range(len(explained_variance)), explained_variance)
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Explained Variance")
    ax.set_title("PCA Explained Variance")

    return explained_variance, fig

def feature_importance(df, target_column):
    """Computes feature importance using Random Forest and plots a bar chart."""
    df_clean = df.dropna()
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Choose classification or regression model
    if y.dtype == 'O' or len(y.unique()) < 10:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        y = LabelEncoder().fit_transform(y)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X, y)
    importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(
        by="Importance", ascending=False
    )

    fig, ax = plt.subplots()
    sns.barplot(data=feature_importance_df, x="Importance", y="Feature", ax=ax)
    ax.set_title("Feature Importance (Random Forest)")

    return feature_importance_df, fig
