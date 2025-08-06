# Plan: TabNet TensorFlow Rebuild

## Objective
Rebuild the TabNet model and pipeline using TensorFlow/Keras for the Telco Customer Churn project, replacing the previous PyTorch implementation.

## Steps
1. Scaffold a new TabNet model class using TensorFlow/Keras in `src/model/tabnet_tf.py`.
2. Update the main pipeline (`src/main.py`) to use the new TensorFlow TabNet model.
3. Ensure compatibility with the existing feature engineering pipeline.
4. Implement training, evaluation, and checkpointing logic for the new model.
5. Integrate SHAP/LIME interpretability if possible with TensorFlow.
6. Update documentation and best practices.
7. Validate outputs and confirm with user.
