"""Step 1 — Register the raw SuperKart data on the Hugging Face Dataset Hub."""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
HF_USERNAME = api.whoami()["name"]
DATASET_REPO = f"{HF_USERNAME}/superkart-sales-dataset"
DATA_DIR = "superkart_project/data"

# Create the dataset repo once; exist_ok makes re-runs safe
api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
print(f"Dataset repo ready: {DATASET_REPO}")

# Upload only the raw file (processed splits are uploaded by prep.py)
api.upload_folder(
    folder_path=DATA_DIR,
    repo_id=DATASET_REPO,
    repo_type="dataset",
    allow_patterns=["SuperKart.csv"],
    commit_message="Register raw SuperKart dataset",
)
print(f"Uploaded raw data -> https://huggingface.co/datasets/{DATASET_REPO}")
