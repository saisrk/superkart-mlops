"""Step 2 — Load raw data from the HF hub, clean it, split it and upload train/test."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
HF_USERNAME = api.whoami()["name"]
DATASET_REPO = f"{HF_USERNAME}/superkart-sales-dataset"
DATA_DIR = "superkart_project/data"
TARGET = "Product_Store_Sales_Total"

# ---------- 1. Load directly from the Hugging Face Dataset Hub ----------
df = pd.read_csv(f"hf://datasets/{DATASET_REPO}/SuperKart.csv")
print("Raw data shape:", df.shape)

# ---------- 2. Clean ----------
df = df.drop_duplicates()

# 'reg' is an inconsistent spelling of 'Regular'
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

# The Product_Id prefix encodes the product category -> keep it as a feature
prefix_map = {"FD": "Food", "NC": "Non-Consumable", "DR": "Drinks"}
df["Product_Category"] = df["Product_Id"].str[:2].map(prefix_map)

# Drop identifiers: Product_Id is unique per row, Store_Id duplicates the store attributes
df = df.drop(columns=["Product_Id", "Store_Id"])
df = df.dropna()
print("Clean data shape:", df.shape)
print("Sugar content levels:", sorted(df["Product_Sugar_Content"].unique()))

# ---------- 3. Train / test split ----------
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["Store_Type"]
)
print(f"Train: {train_df.shape} | Test: {test_df.shape}")

# ---------- 4. Save locally ----------
os.makedirs(DATA_DIR, exist_ok=True)
train_path, test_path = f"{DATA_DIR}/train.csv", f"{DATA_DIR}/test.csv"
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print("Saved:", train_path, test_path)

# ---------- 5. Upload splits back to the Dataset Hub ----------
for path in (train_path, test_path):
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=os.path.basename(path),
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Upload {os.path.basename(path)}",
    )
    print(f"Uploaded {path} -> {DATASET_REPO}")
