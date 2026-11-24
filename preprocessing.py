import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Load dataset
df = pd.read_csv('data/fraud_data_engineered.csv')

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# Remove target and device-related features
X = df.drop(columns=['class', 'device_count', 'shared_device'])
y = df['class']


# Define categorical and numerical columns
categorical_cols = ['source', 'browser', 'sex', 'country']

numerical_cols = [
    'purchase_value',
    'age',
    'account_age',
    'purchase_hour'
]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print("Preprocessing completed!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# Save processed data and preprocessor
os.makedirs('models', exist_ok=True)

joblib.dump(preprocessor, 'models/preprocessor.pkl')

joblib.dump(
    (X_train, X_test, y_train, y_test),
    'models/processed_data.pkl'
)

print("Processed data saved successfully!")
