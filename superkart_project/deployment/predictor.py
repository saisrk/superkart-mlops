"""Scores the portable SuperKart model (plain JSON) with pure Python.

Shared by train.py (to validate the export) and by the Streamlit app, which can then run
without scikit-learn/xgboost - including in the browser."""
import struct


def _f32(x):
    """Round to float32, the precision XGBoost uses for features and thresholds."""
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def predict_row(model, row):
    """Predict Product_Store_Sales_Total for one record (dict of raw feature values)."""
    # 1. Same preprocessing as the pipeline: standard-scale numerics, one-hot encode categoricals
    x = [(row[c] - mu) / sd for c, mu, sd in zip(model["num_cols"], model["mean"], model["scale"])]
    for col, cats in zip(model["cat_cols"], model["categories"]):
        x += [1.0 if row[col] == v else 0.0 for v in cats]      # unknown category -> all zeros
    x = [_f32(v) for v in x]

    # 2. Walk every boosted tree and add up the leaf values
    total = model["base_score"]
    for tree in model["trees"]:
        node = tree[0]
        while "leaf" not in node:
            v = x[node["f"]]
            if model["sparse_input"] and v == 0.0:   # zeros are absent (= missing) in sparse input
                node = tree[node["missing"]]
            else:
                node = tree[node["yes"] if v < _f32(node["t"]) else node["no"]]
        total += node["leaf"]
    return total
