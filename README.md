📌 Automatic Exploratory Data Analysis (EDA) Tool
🚀 A simple and interactive Streamlit app for automatic exploratory data analysis (EDA), including PCA and feature importance analysis.

📖 Overview

This project is a Streamlit-based Exploratory Data Analysis (EDA) tool that allows users to upload CSV datasets and perform various analyses, including:
✅ Data Summary & Statistics
✅ Histograms of Numeric Features
✅ Correlation Matrix Heatmap
✅ Principal Component Analysis (PCA)
✅ Feature Importance using Random Forest

The tool is modularized inside the eda_tool/ package, making it easy to extend and maintain.

📂 Project Structure

Automatic-Exploratory-Data-Analysis/
│── eda_tool/
│   ├── __init__.py
│   ├── data_loader.py          # Handles CSV data loading
│   ├── eda_summary.py          # Provides summary statistics
│   ├── missing_values.py       # Handles missing value analysis
│   ├── outlier_detection.py    # Detects outliers
│   ├── visualization.py        # Visualization functions (histograms, PCA, feature importance)
│── app.py                      # Streamlit app
│── requirements.txt            # Dependencies
│── README.md                   # Project documentation

