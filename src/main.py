"""
TabNet Training Pipeline for Telco Customer Churn

This script loads the processed data, splits it, trains the TabNet model, and evaluates performance.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.model.tabnet_model import TelcoTabNet
import torch
import optuna

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-2, log=True),
        'momentum': trial.suggest_float('momentum', 0.01, 0.4),
        'optimizer_params': dict(lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True)),
    }
    model = TelcoTabNet(seed=42)
    model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val, **params)
    val_results = model.evaluate(X_val, y_val)
    return 1.0 - (val_results['roc_auc'] if val_results['roc_auc'] is not None else 0)

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("data/processed_data.csv")

    # Drop rows with any NaNs
    df = df.dropna()

    # Assume 'Churn' is the target column (adjust if needed)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Use pandas convert_dtypes for robust numeric conversion
    X = X.convert_dtypes()
    # Ensure all columns are numeric (float, int, or bool)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Optuna hyperparameter tuning
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
    print("Best trial:", study.best_trial.params)

    # Train final model with best params
    best_params = study.best_trial.params
    # model = TelcoTabNet(seed=42, device='mps' if torch.backends.mps.is_available() else 'cpu')
    model = TelcoTabNet(seed=42, device='cpu')
    model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val, **best_params)

    # Evaluate
    results = model.evaluate(X_test, y_test)
    print("Test Results:", results)
