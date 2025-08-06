# TabNet Model Integration To-Do List

## Context
This To-Do list tracks the implementation of the TabNet model integration and training pipeline for the Telco Customer Churn project.

## Tasks
- [x] Implement TabNet model class (`src/model/tabnet_model.py`)
- [x] Implement training pipeline script (`src/main.py`)
- [x] Test the full pipeline end-to-end
- [x] Fix TabNetClassifier hyperparameter passing bug
- [x] Disable Optuna, use best params, and retry with MPS backend
- [ ] Tune hyperparameters for best performance
- [ ] Add model interpretability and visualization (feature importance, confusion matrix, ROC curve)
- [ ] Document results and update memory bank

## Next Action
- Run the pipeline with best params and MPS backend, then update memory bank and results.
