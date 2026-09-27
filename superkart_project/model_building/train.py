"""Step 3 — Tune XGBoost, evaluate it and register the best model on the HF Model Hub."""
import os, sys, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)
from xgboost import XGBRegressor
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
HF_USERNAME = api.whoami()["name"]
DATASET_REPO = f"{HF_USERNAME}/superkart-sales-dataset"
MODEL_REPO = f"{HF_USERNAME}/superkart-sales-model"
OUT_DIR = "superkart_project/model_building"
MODEL_FILE = "superkart_sales_model_v1.joblib"
PORTABLE_FILE = "superkart_sales_model_v1_portable.json"
TARGET = "Product_Store_Sales_Total"

# Scoring code shared with the Streamlit app (used here to validate the portable export)
sys.path.insert(0, "superkart_project/deployment")
from predictor import predict_row

# ---------- 1. Load train/test from the Dataset Hub ----------
train_df = pd.read_csv(f"hf://datasets/{DATASET_REPO}/train.csv")
test_df = pd.read_csv(f"hf://datasets/{DATASET_REPO}/test.csv")
X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ---------- 2. Define the model pipeline and the parameter grid ----------
NUM_COLS = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
CAT_COLS = ["Product_Sugar_Content", "Product_Type", "Product_Category",
            "Store_Size", "Store_Location_City_Type", "Store_Type"]

preprocessor = make_column_transformer(
    (StandardScaler(), NUM_COLS),
    (OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
)
xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
model_pipeline = make_pipeline(preprocessor, xgb)

param_grid = {
    "xgbregressor__n_estimators":     [100, 200, 300],
    "xgbregressor__max_depth":        [3, 5, 7],
    "xgbregressor__learning_rate":    [0.05, 0.1],
    "xgbregressor__subsample":        [0.8, 1.0],
    "xgbregressor__colsample_bytree": [0.8, 1.0],
}

# ---------- 3. Tune ----------
grid = GridSearchCV(
    model_pipeline, param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1,
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print(f"Best CV RMSE: {-grid.best_score_:.4f}")

# ---------- 4. Evaluate ----------
def evaluate(model, X, y):
    pred = model.predict(X)
    n, p = X.shape
    r2 = r2_score(y, pred)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2),
        "Adj_R2": float(1 - (1 - r2) * (n - 1) / (n - p - 1)),
        "MAPE": float(mean_absolute_percentage_error(y, pred)),
    }

metrics = {
    "model": "XGBRegressor",
    "best_params": {k.split("__")[1]: v for k, v in grid.best_params_.items()},
    "cv_rmse": float(-grid.best_score_),
    "train": evaluate(best_model, X_train, y_train),
    "test": evaluate(best_model, X_test, y_test),
    "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
}
print(pd.DataFrame({"Train": metrics["train"], "Test": metrics["test"]}).round(4))

# ---------- 5. Save model, portable export, metrics and model card ----------
os.makedirs(OUT_DIR, exist_ok=True)
model_path = f"{OUT_DIR}/{MODEL_FILE}"
joblib.dump(best_model, model_path)

def export_portable(pipeline):
    """Export the fitted pipeline (scaler + encoder + XGBoost trees) to plain JSON,
    so the app can score it without scikit-learn/xgboost (e.g. in the browser)."""
    ct = pipeline.named_steps["columntransformer"]
    scaler = ct.named_transformers_["standardscaler"]
    encoder = ct.named_transformers_["onehotencoder"]
    booster = pipeline.named_steps["xgbregressor"].get_booster()
    cfg = json.loads(booster.save_config())
    base_score = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))
    trees = []
    for dump in booster.get_dump(dump_format="json"):
        nodes, stack = {}, [json.loads(dump)]
        while stack:
            n = stack.pop()
            if "leaf" in n:
                nodes[n["nodeid"]] = {"leaf": n["leaf"]}
            else:
                nodes[n["nodeid"]] = {"f": int(n["split"][1:]), "t": n["split_condition"],
                                      "yes": n["yes"], "no": n["no"], "missing": n["missing"]}
                stack.extend(n["children"])
        trees.append([nodes[i] for i in range(len(nodes))])
    return {"num_cols": NUM_COLS, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "cat_cols": CAT_COLS, "categories": [c.tolist() for c in encoder.categories_],
            "sparse_input": bool(getattr(ct, "sparse_output_", False)),
            "base_score": base_score, "trees": trees}

portable = export_portable(best_model)
# The export must reproduce the pipeline's predictions
check = np.array([predict_row(portable, r) for r in X_test.to_dict("records")])
max_diff = float(np.abs(check - best_model.predict(X_test)).max())
assert max_diff < 0.5, f"Portable export mismatch: {max_diff}"
print(f"Portable export validated on test set (max abs difference {max_diff:.4f})")
portable_path = f"{OUT_DIR}/{PORTABLE_FILE}"
with open(portable_path, "w") as f:
    json.dump(portable, f)
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

t = metrics["test"]
card = f"""---
library_name: sklearn
tags: [xgboost, regression, sales-forecasting, superkart]
---
# SuperKart Sales Forecasting Model

Scikit-learn pipeline (StandardScaler + OneHotEncoder -> XGBRegressor) that predicts
`Product_Store_Sales_Total` for a product in a store.

- `{MODEL_FILE}` - the tuned scikit-learn pipeline (joblib)
- `{PORTABLE_FILE}` - the same model exported to JSON (scaler, encoder, trees) for
  dependency-free scoring; validated to match the pipeline on the test set

Trained on [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO}) at {metrics['trained_at']}.

| Test metric | Value |
|---|---|
| RMSE | {t['RMSE']:.2f} |
| MAE | {t['MAE']:.2f} |
| R2 | {t['R2']:.4f} |
| MAPE | {t['MAPE']:.4f} |

Best hyper-parameters: `{metrics['best_params']}`
"""
card_path = f"{OUT_DIR}/MODEL_CARD.md"
with open(card_path, "w") as f:
    f.write(card)

# ---------- 6. Register the best model on the HF Model Hub ----------
api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, exist_ok=True)
for local, remote in [(model_path, MODEL_FILE),
                      (portable_path, PORTABLE_FILE),
                      (f"{OUT_DIR}/metrics.json", "metrics.json"),
                      (card_path, "README.md")]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=MODEL_REPO, repo_type="model",
                    commit_message=f"Register {remote}")
print(f"Best model registered -> https://huggingface.co/{MODEL_REPO}")
