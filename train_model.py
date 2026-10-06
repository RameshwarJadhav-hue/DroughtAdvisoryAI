"""
Machine Learning Training Script for Drought Risk Prediction
Trains a Random Forest Classifier on agro-meteorological features
and saves the trained model to disk for fast inference in the web app.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate():
    data_path = os.path.join(os.path.dirname(__file__), "drought_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    print(f"--> Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target distribution:\n{df['Drought_Risk'].value_counts()}\n")

    features = [
        "Rainfall_mm",
        "Normal_Rainfall_mm",
        "Rainfall_Deficit_pct",
        "Soil_Moisture_pct",
        "Groundwater_Level_m",
        "Avg_Temp_C"
    ]

    X = df[features]
    y = df["Drought_Risk"]

    # Stratified split to ensure all classes (Low, Medium, High) are represented in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 1. Baseline Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    print(f"Decision Tree Baseline Accuracy: {dt_acc * 100:.2f}%")

    # 2. Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Classifier Accuracy: {rf_acc * 100:.2f}%")

    print("\n--- Classification Report (Random Forest) ---")
    print(classification_report(y_test, rf_pred, zero_division=0))

    print("--- Confusion Matrix ---")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, rf_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(cm_df)

    # Feature Importance Analysis
    print("\n--- Feature Importance Ranking ---")
    importances = rf_model.feature_importances_
    for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<22}: {imp * 100:5.2f}%")

    # Fit final model on all data for the interactive app
    final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    final_model.fit(X, y)

    model_path = os.path.join(os.path.dirname(__file__), "drought_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\n--> Trained model successfully exported to: {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
