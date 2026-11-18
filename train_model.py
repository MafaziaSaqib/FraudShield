
import pandas as pd
import joblib
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load processed data
X_train, X_test, y_train, y_test = joblib.load(
    'models/processed_data.pkl'
)

# 1. Decision Tree Classifier

dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced'
)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

print("\n--- DECISION TREE ---")

print("Accuracy:", accuracy_score(y_test, dt_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    dt_predictions,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_predictions))

joblib.dump(dt_model, 'models/decision_tree.pkl')


# 2. Random Forest Classifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\n--- RANDOM FOREST ---")

print("Accuracy:", accuracy_score(y_test, rf_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_predictions,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

joblib.dump(rf_model, 'models/random_forest.pkl')


# 3. K-Means Clustering

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

# Train K-Means on training data
clusters = kmeans.fit_predict(X_train)

print("\n--- K-MEANS CLUSTERING ---")

# Compare clusters with actual fraud labels
cluster_analysis = pd.DataFrame({
    'Cluster': clusters,
    'Actual Class': y_train.reset_index(drop=True)
})

print("\nCluster Distribution:")
print(cluster_analysis['Cluster'].value_counts().sort_index())

print("\nCluster vs Actual Fraud:")
print(pd.crosstab(
    cluster_analysis['Cluster'],
    cluster_analysis['Actual Class']
))

# Calculate fraud percentage in each cluster
print("\nFraud Percentage in Each Cluster:")

cluster_fraud_rate = cluster_analysis.groupby('Cluster')['Actual Class'].mean() * 100

print(cluster_fraud_rate)

joblib.dump(kmeans, 'models/k_means.pkl')


print("\nAll three models trained and saved successfully!")


from sklearn.metrics import precision_score, recall_score, f1_score

# Get fraud probabilities
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

# Test different prediction thresholds
thresholds = [0.5, 0.4, 0.3, 0.2]

print("\n--- RANDOM FOREST THRESHOLD COMPARISON ---")

for threshold in thresholds:

    predictions = (rf_probabilities >= threshold).astype(int)

    print("\nThreshold:", threshold)

    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1-Score:", f1_score(y_test, predictions, zero_division=0))


from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# Evaluate both classification models
results = []

for name, predictions in [
    ('Decision Tree', dt_predictions),
    ('Random Forest', rf_predictions)
]:

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, zero_division=0),
        'Recall': recall_score(y_test, predictions, zero_division=0),
        'F1 Score': f1_score(y_test, predictions, zero_division=0)
    })

results_df = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ---")
print(results_df)

results_df.to_csv('models/model_comparison.csv', index=False)

# Save K-Means cluster analysis
cluster_analysis.to_csv(
    'models/cluster_analysis.csv',
    index=False
)

print("\nEvaluation results saved successfully!")
