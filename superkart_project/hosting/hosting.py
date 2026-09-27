"""Step 4 — Create the Hugging Face Space and push all deployment files to it."""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
HF_USERNAME = api.whoami()["name"]
SPACE_REPO = f"{HF_USERNAME}/superkart-sales-forecaster"
MODEL_REPO = f"{HF_USERNAME}/superkart-sales-model"
DEPLOY_DIR = "superkart_project/deployment"

# Create a public Space (no-op if it already exists). Static Spaces are free;
# Docker Spaces need a PRO plan on free hardware — the same files work with sdk "docker".
api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="static",
                private=False, exist_ok=True)

# Push Dockerfile, requirements.txt, predictor.py, index.html and README.md
api.upload_folder(folder_path=DEPLOY_DIR, repo_id=SPACE_REPO, repo_type="space",
                  ignore_patterns=["app.py", "__pycache__/*"],
                  commit_message="Deploy SuperKart Streamlit app")

# Push app.py with this account's model repo filled in (also set for Docker via the Dockerfile)
for name in ("app.py", "Dockerfile"):
    with open(f"{DEPLOY_DIR}/{name}") as f:
        content = f.read().replace("your-hf-username/superkart-sales-model", MODEL_REPO)
    api.upload_file(path_or_fileobj=content.encode(), path_in_repo=name,
                    repo_id=SPACE_REPO, repo_type="space",
                    commit_message=f"Deploy {name} (model: {MODEL_REPO})")

print(f"Deployment files pushed -> https://huggingface.co/spaces/{SPACE_REPO}")
print("Files in Space:", api.list_repo_files(SPACE_REPO, repo_type="space"))
