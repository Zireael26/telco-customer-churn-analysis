"""
TabNet Model Integration for Telco Customer Churn

This module provides a class to initialize, train, and evaluate a TabNetClassifier using the pytorch-tabnet2 library.
"""
import numpy as np
import pandas as pd
from pytorch_tabnet import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torch

class TelcoTabNet:
    def load(self, model_path, **kwargs):
        """Load a TabNet model from a checkpoint."""
        # Use same device and seed as current instance
        self.model = TabNetClassifier(seed=self.seed, device_name=self.device, **kwargs)
        self.model.load_model(model_path)
    def __init__(self, seed: int = 42, device: str = None):
        self.seed = seed
        # Always respect explicit device argument
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model = None

    def fit(self, X, y, X_valid=None, y_valid=None, **kwargs):
        np.random.seed(self.seed)
        self.model = TabNetClassifier(seed=self.seed, device_name=self.device, **kwargs)
        self.model.fit(
            X_train=X.values,
            y_train=y.values,
            eval_set=[(X_valid.values, y_valid.values)] if X_valid is not None and y_valid is not None else None,
            eval_name=["val"] if X_valid is not None and y_valid is not None else None,
            eval_metric=["accuracy"],
            max_epochs=100,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return self.model.predict(X.values)
        return self.model.predict(X)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            return self.model.predict_proba(X.values)
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        proba = self.predict_proba(X)[:, 1] if self.model.output_dim == 2 else None
        acc = accuracy_score(y, preds)
        roc = roc_auc_score(y, proba) if proba is not None else None
        cm = confusion_matrix(y, preds)
        return {"accuracy": acc, "roc_auc": roc, "confusion_matrix": cm}
