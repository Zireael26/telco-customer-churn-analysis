"""
TabNet Training Pipeline for Telco Customer Churn

This script loads the processed data, splits it, trains the TabNet model, and evaluates performance.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from src.model.tabnet_tf import TabNet
from tensorflow import keras

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'n_glu': trial.suggest_int('n_glu', 1, 4),
        'decision_dim': trial.suggest_int('decision_dim', 8, 64),
        'relaxation_factor': trial.suggest_float('relaxation_factor', 1.0, 2.0),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
    }
    feature_dim = X_train.shape[1]
    output_dim = len(set(y_train))
    model = TabNet(feature_dim=feature_dim, output_dim=output_dim, **{k: params[k] for k in params if k not in ['lr']})
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train.values, y_train.values, validation_data=(X_val.values, y_val.values), epochs=20, batch_size=256, verbose=0)
    eval_results = model.evaluate(X_val.values, y_val.values, verbose=0)
    return 1.0 - eval_results[1]  # maximize accuracy

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("data/processed_data.csv")

    # Drop rows with any NaNs
    df = df.dropna()


    # Assume 'Churn' is the target column (adjust if needed)
    y = df["Churn"]
    # Drop Churn and any Churn-derived columns from features
    churn_cols = [col for col in df.columns if col.startswith("Churn")]
    X = df.drop(churn_cols, axis=1)

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



    # --- Optuna Hyperparameter Tuning ---
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
    print("Best trial:", study.best_trial.params)
    best_params = study.best_trial.params
    feature_dim = X_train.shape[1]
    output_dim = len(y_train.unique())
    model = TabNet(feature_dim=feature_dim, output_dim=output_dim,
                  n_steps=best_params.get('n_steps', 3),
                  n_glu=best_params.get('n_glu', 2),
                  decision_dim=best_params.get('decision_dim', 8),
                  relaxation_factor=best_params.get('relaxation_factor', 1.5))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_params.get('lr', 0.001)), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Training TensorFlow TabNet model with best Optuna params...")
    history = model.fit(X_train.values, y_train.values, validation_data=(X_val.values, y_val.values), epochs=50, batch_size=256)
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save(os.path.join(checkpoint_dir, "tabnet_tf_model.keras"))
    print(f"Model checkpoint saved to {os.path.join(checkpoint_dir, 'tabnet_tf_model.keras')}")

    # Evaluate
    eval_results = model.evaluate(X_test.values, y_test.values, verbose=2)
    print("Test Results:", dict(zip(model.metrics_names, eval_results)))

    # --- Model Interpretability and Visualization ---
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc

    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Feature Importance (TabNet)
    # TensorFlow TabNet does not provide feature_importances_ directly.
    print('TensorFlow TabNet feature importances not available. Use SHAP/LIME for interpretability.')

    # 2. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_test.values)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test.values, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "confusion_matrix.png"))

    # 3. ROC Curve
    y_score = y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred[:, 0]
    fpr, tpr, thresholds = roc_curve(y_test.values, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "roc_curve.png"))

    # SHAP interpretability disabled for speed.

    # --- LIME Feature Importance for TabNet ---
    try:
        import lime
        from lime.lime_tabular import LimeTabularExplainer
        X_lime = X_test.sample(n=min(200, len(X_test)), random_state=42)
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns.tolist(),
            class_names=["No Churn", "Churn"],
            mode="classification"
        )
        i = 0
        exp = explainer.explain_instance(
            data_row=np.array(X_lime.iloc[i]),
            predict_fn=lambda x: model.predict(x),
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
