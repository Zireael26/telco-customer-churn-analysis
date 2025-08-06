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

    # Only cast columns that are already numeric to float32
    for df_ in [X_train, X_val, X_test]:
        for col in df_.columns:
            if pd.api.types.is_numeric_dtype(df_[col]):
                df_[col] = df_[col].astype('float32')



    # --- Model Checkpoint Selection/Training ---
    import glob
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.zip")))
    print("\nAvailable model checkpoints:")
    for idx, ckpt in enumerate(checkpoint_files):
        print(f"  [{idx}] {os.path.basename(ckpt)}")
    print(f"  [n] Train a new model")
    user_choice = input("Select checkpoint index to load, or 'n' to train new: ").strip()

    best_params = {
        'n_d': 18,
        'n_a': 46,
        'n_steps': 5,
        'gamma': 1.5377185972039231,
        'lambda_sparse': 2.4791381958764685e-06,
        'momentum': 0.31119131379716436,
        'optimizer_params': {'lr': 0.0013782495262724347}
    }
    device = 'cpu'
    if user_choice.isdigit() and int(user_choice) < len(checkpoint_files):
        model_path = checkpoint_files[int(user_choice)]
        print(f"Loading model from {model_path}")
        model = TelcoTabNet(seed=42, device=device)
        fit_params = best_params.copy()
        fit_params.pop('optimizer_params', None)
        model.load(model_path, **fit_params)
        trained = True
    else:
        print("Training new model...")
        print(f"Using device: {device}")
        model = TelcoTabNet(seed=42, device=device)
        fit_params = best_params.copy()
        fit_params.pop('optimizer_params', None)
        model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val, optimizer_params=best_params['optimizer_params'], **fit_params)
        model_path = os.path.join(checkpoint_dir, "tabnet_model.zip")
        model.model.save_model(model_path)
        print(f"Model checkpoint saved to {model_path}")
        trained = True

    # Evaluate
    results = model.evaluate(X_test, y_test)
    print("Test Results:", results)

    # --- Model Interpretability and Visualization ---
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc

    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Feature Importance (TabNet)
    if hasattr(model.model, 'feature_importances_'):
        importances = model.model.feature_importances_
        feature_names = X_test.columns
        sorted_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        sns.barplot(x=importances[sorted_idx][:20], y=np.array(feature_names)[sorted_idx][:20])
        plt.title('Top 20 TabNet Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "tabnet_feature_importance.png"))
        # plt.show()
    else:
        print('TabNet feature importances not available.')

    # 2. Confusion Matrix
    cm = results['confusion_matrix']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "confusion_matrix.png"))
    # plt.show()

    # 3. ROC Curve
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "roc_curve.png"))
    # plt.show()

    # --- SHAP Feature Importance for TabNet ---
    try:
        import shap
        # Use a small sample for SHAP to save time
        X_shap = X_test.sample(n=min(200, len(X_test)), random_state=42)
        # Use the correct predict_proba function for SHAP
        def predict_fn(X):
            # Always pass X directly; model.predict_proba expects numpy array
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            return model.predict_proba(X)
        explainer = shap.KernelExplainer(predict_fn, X_shap)
        shap_values = explainer.shap_values(X_shap, nsamples=100)
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (TabNet)")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "shap_feature_importance.png"))
        plt.close()
        print("SHAP feature importance plot saved.")
    except ImportError:
        print("shap library not installed. Run 'pip install shap' to enable SHAP interpretability.")
    except Exception as e:
        print(f"SHAP interpretability failed: {e}")

    # --- LIME Feature Importance for TabNet ---
    try:
        import lime
        from lime.lime_tabular import LimeTabularExplainer
        # Use a small sample for LIME to save time
        X_lime = X_test.sample(n=min(200, len(X_test)), random_state=42)
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns.tolist(),
            class_names=["No Churn", "Churn"],
            mode="classification"
        )
        # Explain a single instance (first in X_lime)
        i = 0
        exp = explainer.explain_instance(
            data_row=np.array(X_lime.iloc[i]),
            predict_fn=lambda x: model.predict_proba(x),
            num_features=20
        )
        fig = exp.as_pyplot_figure()
        plt.title("LIME Feature Importance (TabNet)")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "lime_feature_importance.png"))
        plt.close()
        print("LIME feature importance plot saved.")
    except ImportError:
        print("lime library not installed. Run 'pip install lime' to enable LIME interpretability.")
    except Exception as e:
        print(f"LIME interpretability failed: {e}")

    # Example: Load model and predict on new data
    # loaded_model = TabNetClassifier()
    # loaded_model.load_model(model_path)
    # preds = loaded_model.predict(X_test.values)
    # print('Loaded model predictions:', preds[:10])
