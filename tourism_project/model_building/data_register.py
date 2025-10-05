from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "Yash0204/prediction-tourism-mlops"
repo_type = "dataset"
data_path = "tourism_project/data"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Ensure folder exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Data folder not found at {data_path}")

# Ensure repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"ℹ️ Repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"✅ Repo '{repo_id}' created.")

# Upload data folder
try:
    files = os.listdir(data_path)
    print(f"📦 Uploading {len(files)} files to {repo_id} ...")
    api.upload_folder(folder_path=data_path, repo_id=repo_id, repo_type=repo_type)
    print(f"🎉 Upload complete! View dataset at: https://huggingface.co/datasets/{repo_id}")
except Exception as e:
    print(f"⚠️ Upload failed: {e}")
