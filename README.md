# FraudShield — Online Shopping Fraud Detection

## Project Overview

FraudShield is a machine learning project developed to analyze online shopping transactions and identify potentially fraudulent activities.

The project uses transaction-related information to classify transactions as normal or fraudulent and explores transaction patterns through clustering.

## Objectives

- Analyze online shopping transaction data.
- Identify potentially fraudulent transactions.
- Compare Decision Tree and Random Forest classification models.
- Explore transaction patterns using K-Means clustering.
- Present insights and predictions through an interactive Streamlit dashboard.

## Machine Learning Algorithms

1. **Decision Tree:** Supervised classification algorithm used to classify transactions.
2. **Random Forest:** Ensemble classification algorithm used to improve prediction performance.
3. **K-Means:** Unsupervised clustering algorithm used to explore transaction patterns.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure

```text
FraudShield/
├── data/
├── notebooks/
├── models/
├── app.py
├── preprocessing.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The project uses the Fraud Ecommerce dataset and the IP Address to Country dataset.

Dataset source:
https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce

The datasets are used for transaction analysis, feature engineering, and model development.

## Installation and Setup

1. Clone this repository:

   `git clone YOUR_REPOSITORY_URL`

2. Open the project folder:

   `cd FraudShield`

3. Create a virtual environment:

   `python -m venv .venv`

4. Activate the environment:

   `.\.venv\Scripts\activate`

5. Install the required libraries:

   `pip install -r requirements.txt`

6. Ensure the required dataset and trained model files are available in their respective folders.

7. Run the dashboard:

   `python -m streamlit run app.py`

## Dashboard Features

- Project overview and transaction statistics.
- Fraud detection using trained classification models.
- K-Means transaction cluster analysis.
- Interactive charts and visualizations.

## Project Status

Academic Machine Learning project developed as part of the BS Artificial Intelligence program.
