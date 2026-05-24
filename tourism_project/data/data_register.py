"""Upload the raw tourism.csv to the Hugging Face Dataset Hub."""
import os
from huggingface_hub import HfApi, create_repo, login

HF_USERNAME     = "raghavv33"          # <-- change me
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-dataset"
LOCAL_CSV_PATH  = "tourism_project/data/tourism.csv"

def main():
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)
    api = HfApi(token=hf_token)
    create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset",
                exist_ok=True, token=hf_token)
    print(f"Dataset repo ready: {DATASET_REPO_ID}")
    api.upload_file(path_or_fileobj=LOCAL_CSV_PATH,
                    path_in_repo="tourism.csv",
                    repo_id=DATASET_REPO_ID, repo_type="dataset")
    print(f"Uploaded -> {DATASET_REPO_ID}/tourism.csv")

if __name__ == "__main__":
    main()
