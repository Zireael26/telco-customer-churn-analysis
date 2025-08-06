"""
TabNet Training Pipeline for Telco Customer Churn

This script loads the processed data, splits it, trains the TabNet model, and evaluates performance.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model.tabnet_model import TelcoTabNet

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("data/processed_data.csv")

    # Assume 'Churn' is the target column (adjust if needed)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Initialize and train TabNet
    model = TelcoTabNet(seed=42)
    model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)

    # Evaluate
    results = model.evaluate(X_test, y_test)
    print("Test Results:", results)
