import os
import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

os.makedirs("models", exist_ok=True)

X_train, X_test, y_train, y_test = joblib.load(
    "models/processed_data.pkl"
)

y_train_array = np.asarray(y_train).ravel()
y_test_array = np.asarray(y_test).ravel()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training class distribution:")
print(pd.Series(y_train_array).value_counts().sort_index())

if 1 not in np.unique(y_train_array):
    raise ValueError(
        "Fraud class (1) is missing from training labels. "
        "Check preprocessing.py and the train-test split."
    )

# DECISION TREE CLASSIFIER

dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10
)

dt_model.fit(X_train, y_train_array)

dt_predictions = dt_model.predict(X_test)

print("\n--- DECISION TREE ---")
print("Accuracy:", accuracy_score(y_test_array, dt_predictions))
print("\nClassification Report:")
print(classification_report(
    y_test_array,
    dt_predictions,
    zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(y_test_array, dt_predictions))

print("Decision Tree classes:", dt_model.classes_)
dt_probabilities = dt_model.predict_proba(X_test)
print("Decision Tree probability output shape:", dt_probabilities.shape)

joblib.dump(dt_model, "models/decision_tree.pkl")

# RANDOM FOREST CLASSIFIER

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train_array)

rf_predictions = rf_model.predict(X_test)

print("\n--- RANDOM FOREST ---")
print("Accuracy:", accuracy_score(y_test_array, rf_predictions))
print("\nClassification Report:")
print(classification_report(
    y_test_array,
    rf_predictions,
    zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(y_test_array, rf_predictions))

joblib.dump(rf_model, "models/random_forest.pkl")

# K-MEANS CLUSTERING

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_train)

print("\n--- K-MEANS CLUSTERING ---")

cluster_analysis = pd.DataFrame({
    "Cluster": clusters,
    "Actual Class": y_train_array
})

print("\nCluster Distribution:")
print(cluster_analysis["Cluster"].value_counts().sort_index())

print("\nCluster vs Actual Fraud:")
print(pd.crosstab(
    cluster_analysis["Cluster"],
    cluster_analysis["Actual Class"]
))

print("\nFraud Percentage in Each Cluster:")
cluster_fraud_rate = (
    cluster_analysis.groupby("Cluster")["Actual Class"].mean() * 100
)
print(cluster_fraud_rate)

joblib.dump(kmeans, "models/k_means.pkl")

# RANDOM FOREST THRESHOLD COMPARISON

rf_fraud_index = np.where(rf_model.classes_ == 1)[0]

if len(rf_fraud_index) == 0:
    raise ValueError("Fraud class (1) not found in Random Forest classes.")

rf_probabilities = rf_model.predict_proba(X_test)[:, rf_fraud_index[0]]

thresholds = [0.5, 0.4, 0.3, 0.2]

print("\n--- RANDOM FOREST THRESHOLD COMPARISON ---")

for threshold in thresholds:
    threshold_predictions = (
        rf_probabilities >= threshold
    ).astype(int)

    print("\nThreshold:", threshold)
    print("Precision:", precision_score(
        y_test_array, threshold_predictions, zero_division=0
    ))
    print("Recall:", recall_score(
        y_test_array, threshold_predictions, zero_division=0
    ))
    print("F1-Score:", f1_score(
        y_test_array, threshold_predictions, zero_division=0
    ))

# MODEL COMPARISON

results = []

for name, predictions in [
    ("Decision Tree", dt_predictions),
    ("Random Forest", rf_predictions)
]:
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_array, predictions),
        "Precision": precision_score(
            y_test_array, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test_array, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test_array, predictions, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ---")
print(results_df)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

# SAVE K-MEANS CLUSTER ANALYSIS

cluster_analysis.to_csv(
    "models/cluster_analysis.csv",
    index=False
)

print("\nAll three models trained and saved successfully!")