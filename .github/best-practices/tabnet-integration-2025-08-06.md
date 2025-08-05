# Python Project Best Practices for TabNet Integration

## Directory Structure

- `data/` – Raw and processed datasets (never commit large/raw data).
- `notebooks/` – Jupyter notebooks for exploration and prototyping.
- `src/`
  - `main.py` – Main entry point for training/evaluation.
  - `engineer/` – Feature engineering modules.
  - `model/` – Model definition and training scripts.
  - `preprocess/` – Data preprocessing scripts.
- `.github/` – Project management, plans, and best-practices.
- `requirements.txt` – List of dependencies.
- `README.md` – Project overview and instructions.
- `tests/` – Unit and integration tests (recommended).

## Coding Standards & Conventions

- **Code Organization**: Keep code modular (separate preprocessing, feature engineering, modeling, and evaluation).
- **Documentation**: Use clear docstrings and inline comments.
- **Naming**: Use descriptive variable and function names.
- **Error Handling**: Use try/except blocks where appropriate.
- **Reproducibility**: Set random seeds for all libraries (numpy, torch, etc.).
- **Version Control**: Commit changes with meaningful messages.
- **Testing**: Write unit and integration tests for all new code.
- **Data Handling**: Never commit raw data; use `.gitignore` for large files.
- **Dependency Management**: Keep `requirements.txt` updated.
- **Interpretability**: Always provide feature importance or model explainability outputs.
- **Performance**: Use efficient data loading and batching.
- **Visualization**: Generate and save as many visualizations as possible for data exploration, feature engineering, model evaluation, and interpretability.
- **Environment Management**: Always check for and activate the correct Python virtual environment before running scripts.
- **Hardware Utilization**: Detect and use GPU if available for PyTorch; otherwise, use CPU.
- **Notebook Hygiene**: Keep notebooks clean, with outputs cleared before committing.
- **Logging**: Use logging for important events, warnings, and errors.
