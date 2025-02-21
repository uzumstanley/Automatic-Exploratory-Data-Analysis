link to the project:  https://automatic-exploratory-data-analysis-rfzmcrpraqxdvfnuuyms6k.streamlit.app

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


🚀 Features

🔹 Data Summary & Insights
Displays key statistics such as mean, median, and standard deviation.
Identifies missing values in the dataset.
🔹 Data Visualization
Histograms for numerical columns.
Correlation Matrix heatmap for feature relationships.
🔹 Advanced Analysis
Principal Component Analysis (PCA): Visualizes explained variance.
Feature Importance Analysis: Uses Random Forest to determine the most important features.
💻 Installation & Setup

1️⃣ Clone the repository
git clone https://github.com/your-username/Automatic-Exploratory-Data-Analysis.git
cd Automatic-Exploratory-Data-Analysis
2️⃣ Create a virtual environment (Optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run the Streamlit app
streamlit run app.py
📌 Usage

Open the Streamlit interface in your browser.
Upload a CSV file.
Explore the dataset using the available analysis options.
Select a target variable for feature importance analysis.
🔧 Configuration

🛠️ Technologies Used

Python 3.x
Pandas, NumPy (Data Processing)
Matplotlib, Seaborn (Visualizations)
Scikit-learn (Machine Learning for PCA & Feature Importance)
Streamlit (Interactive Web App)
📌 Future Improvements

✅ Support for time-series analysis.
✅ More outlier detection techniques.
✅ Dynamic feature selection for ML model training.

