---
library_name: sklearn
tags: [xgboost, regression, sales-forecasting, superkart]
---
# SuperKart Sales Forecasting Model

Scikit-learn pipeline (StandardScaler + OneHotEncoder -> XGBRegressor) that predicts
`Product_Store_Sales_Total` for a product in a store.

- `superkart_sales_model_v1.joblib` - the tuned scikit-learn pipeline (joblib)
- `superkart_sales_model_v1_portable.json` - the same model exported to JSON (scaler, encoder, trees) for
  dependency-free scoring; validated to match the pipeline on the test set

Trained on [saisrk/superkart-sales-dataset](https://huggingface.co/datasets/saisrk/superkart-sales-dataset) at 2026-09-27 09:54:26 UTC.

| Test metric | Value |
|---|---|
| RMSE | 270.66 |
| MAE | 104.56 |
| R2 | 0.9360 |
| MAPE | 0.0381 |

Best hyper-parameters: `{'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 7, 'n_estimators': 100, 'subsample': 0.8}`
