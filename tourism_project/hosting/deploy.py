"""Push the Streamlit deployment files to a Hugging Face Space."""
import os
from huggingface_hub import HfApi, create_repo, login

HF_USERNAME   = "raghavv33"           # <-- change me
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-package-app"
DEPLOY_DIR    = "tourism_project/deployment"

def main():
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)
    api = HfApi(token=hf_token)
    create_repo(repo_id=SPACE_REPO_ID, repo_type="space",
                space_sdk="docker", exist_ok=True, token=hf_token)
    print(f"Space ready: {SPACE_REPO_ID}")
    api.upload_folder(folder_path=DEPLOY_DIR, repo_id=SPACE_REPO_ID,
                      repo_type="space",
                      commit_message="Deploy updated tourism app")
    print(f"App URL: https://huggingface.co/spaces/{SPACE_REPO_ID}")

if __name__ == "__main__":
    main()
