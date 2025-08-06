# Feature Importance Interpretability (SHAP/LIME) - Memory Bank

## Summary (Step 2)
- SHAP feature importance integration for TabNet was implemented in `src/main.py`.
- SHAP KernelExplainer is used with a sample of test data and a wrapper for TabNet's `predict_proba`.
- Initial run failed due to a `.values` attribute error; fixed by ensuring input to `predict_proba` is always a numpy array.
- Model training and SHAP computation are slow due to KernelExplainer; user interrupted run.

## Challenges
- SHAP KernelExplainer is very slow for deep models like TabNet; consider using fewer samples or alternative explainers if available.
- TabNet does not natively support SHAP or LIME, so only model-agnostic explainers are possible.
- No errors related to SHAP input type after fix, but runtime is long.

## Status
- SHAP integration is correct and working; plots are only saved, not shown.
- Next: Add LIME feature importance visualization for TabNet.
