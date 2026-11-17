import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Load data
data = pd.read_csv("dataset/TARP.csv")
data.columns = data.columns.str.strip().str.lower()

# Select features & clean 
data = data[['soil moisture', 'temperature', 'soil humidity',
             'air temperature (c)', 'status']].dropna()

data['status'] = data['status'].map({'ON': 1, 'OFF': 0})

print("Shape after cleaning:", data.shape)
print("Null values:\n", data.isnull().sum())
print("Duplicates:", data.duplicated().sum())

# Feature engineering 
data['moisture_temp']     = data['soil moisture'] * data['temperature']
data['moisture_humidity'] = data['soil moisture'] * data['soil humidity']

# Define X and y 
FEATURES = ['soil moisture', 'temperature', 'soil humidity',
            'air temperature (c)', 'moisture_temp', 'moisture_humidity']

X = data[FEATURES]
y = data['status']

print("\nX shape:", X.shape)
print("Class distribution:\n", y.value_counts())

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_scaled, y_train)
print("\nModel training complete")

# Evaluate
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

importance = pd.Series(model.feature_importances_, index=FEATURES)
print("\nFeature Importance:\n", importance.sort_values(ascending=False))

# Save model & scaler
os.makedirs("model", exist_ok=True)

with open("model/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel and scaler saved to model/ folder")
