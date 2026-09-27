"""Step 4 — Create the Hugging Face Space and push all deployment files to it."""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
HF_USERNAME = api.whoami()["name"]
SPACE_REPO = f"{HF_USERNAME}/superkart-sales-forecaster"
MODEL_REPO = f"{HF_USERNAME}/superkart-sales-model"

# Create a public Docker Space (no-op if it already exists)
api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker",
                private=False, exist_ok=True)

# Tell the app which model repo to load
api.add_space_variable(repo_id=SPACE_REPO, key="MODEL_REPO", value=MODEL_REPO)

# Push Dockerfile, app.py, requirements.txt and README.md
api.upload_folder(
    folder_path="superkart_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    commit_message="Deploy SuperKart Streamlit app",
)
print(f"Deployment files pushed -> https://huggingface.co/spaces/{SPACE_REPO}")
