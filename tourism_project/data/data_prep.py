"""Clean the data, split it, and push train/test to HF."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download, login

HF_USERNAME      = "raghavv33"           # <-- change me
DATASET_REPO_ID  = f"{HF_USERNAME}/tourism-dataset"
LOCAL_DIR        = "tourism_project/data"
TARGET_COL       = "ProdTaken"
TEST_SIZE        = 0.2
RANDOM_STATE     = 42

def clean(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    if "NumberOfTrips" in df.columns:
        med = df["NumberOfTrips"].median()
        df.loc[df["NumberOfTrips"] > 12, "NumberOfTrips"] = med
    for col in df.select_dtypes(include=["number"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df.drop_duplicates().reset_index(drop=True)

def main():
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)
    raw = hf_hub_download(repo_id=DATASET_REPO_ID, filename="tourism.csv",
                          repo_type="dataset", token=hf_token)
    df = pd.read_csv(raw)
    print(f"Loaded raw dataset: {df.shape}")
    df_clean = clean(df)
    print(f"Cleaned: {df_clean.shape}")
    train_df, test_df = train_test_split(df_clean, test_size=TEST_SIZE,
                                        stratify=df_clean[TARGET_COL],
                                        random_state=RANDOM_STATE)
    print(f"Train {train_df.shape} Test {test_df.shape}")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    train_df.to_csv(f"{LOCAL_DIR}/train.csv", index=False)
    test_df.to_csv(f"{LOCAL_DIR}/test.csv", index=False)
    api = HfApi(token=hf_token)
    for fname in ["train.csv", "test.csv"]:
        api.upload_file(path_or_fileobj=f"{LOCAL_DIR}/{fname}",
                        path_in_repo=fname,
                        repo_id=DATASET_REPO_ID, repo_type="dataset")
        print(f"Uploaded {fname}")

if __name__ == "__main__":
    main()
