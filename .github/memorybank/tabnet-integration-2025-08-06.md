# Memory Bank: TabNet Integration

## Feature: Data Exploration

- Date: 2025-08-06
- Status: Complete
- Location: notebooks/data_exploration.ipynb

### Summary
- Created and executed a data exploration notebook for the Telco Customer Churn dataset.
- Explored data shape, columns, types, distributions, missing values, and correlations.
- No missing values found; visualizations generated for both numerical and categorical features.
- Notebook reviewed and approved by user.

### To-Do List (Feature Engineering Next Steps)
- Plan and implement feature engineering based on exploration insights.
- Document new features and rationale in memory bank.
- Update pipeline and notebook as features are added.

---



## Feature: Feature Engineering Script Implementation

- Date: 2025-08-06
- Status: Script implemented and tested successfully
- Location: src/engineer/feature_engineering.py

### Summary
- Implemented feature engineering as a modular, well-documented script based on validated notebook logic.
- Script includes:
  - Dropping/encoding 'customerID'
  - Converting 'TotalCharges' to numeric
  - Encoding categorical variables (label encoding for binary, one-hot for multi-class)
  - Creating tenure group bins and one-hot encoding them
  - Advanced features: interaction (MonthlyCharges * tenure), log transform (TotalCharges), binning (MonthlyCharges)
- Script tested with the dataloader and sample data; output file generated successfully with no errors.

### Rationale
- Each feature engineering step is designed to improve model interpretability and performance:
  - Dropping IDs and converting types ensures clean input.
  - Encoding categoricals and binning captures non-linearities and class structure.
  - New features (interactions, log transforms) help TabNet learn complex relationships.

### Challenges
- Ensured all notebook logic was faithfully translated to script, with modular functions and docstrings.
- Confirmed pandas/numpy imports; environment is set up correctly.


### Reproducibility & Robustness
- The feature engineering script was run multiple times on the same input data, producing identical outputs each time.
- This confirms the pipeline is reproducible and robust for the current dataset and environment.


## Feature: TabNet Model Integration & Training Pipeline

- Date: 2025-08-06
- Status: Initial implementation complete
- Location: src/model/tabnet_model.py, src/main.py

### Summary
- Created `TelcoTabNet` class for TabNetClassifier initialization, training, and evaluation.
- Implemented `src/main.py` to load processed data, split into train/val/test, train TabNet, and evaluate on test set.
- Pipeline is ready for further testing, hyperparameter tuning, and interpretability steps.

### Recent Actions (2025-08-06)
- Disabled Optuna, used best params from previous tuning.
- Updated pipeline to pass only valid arguments to TabNetClassifier.
- Set device to 'mps' if available, else fallback to 'cpu'.
- Ran pipeline with MPS backend enabled.

### Challenges
- MPS backend is detected and used, but TabNet/pytorch-tabnet2 triggers a PyTorch internal assertion and segmentation fault on MPS (as_strided_tensorimpl does not work with MPS).
- This is a known PyTorch/MPS limitation; not fixable at the user code level.

### Current Status
- Pipeline runs and trains successfully on CPU.
- MPS backend is not supported for TabNetClassifier due to PyTorch bug. All training will use CPU for stability.
- MPS investigation is closed; limitation documented.
- Next: Add model interpretability and visualization (feature importance, confusion matrix, ROC curve).

---

### Next Steps
- Revert to CPU backend for TabNet training.
- Add model interpretability and visualization (feature importance, confusion matrix, ROC curve).
- Document results and update memory bank.

This memory bank entry will be updated as the TabNet integration progresses.
