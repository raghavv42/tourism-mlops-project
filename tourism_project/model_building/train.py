"""Train + tune XGBoost, log to MLflow, push model to HF Model Hub."""
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, create_repo, login
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

HF_USERNAME     = "raghavv33"           # <-- change me
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-dataset"
MODEL_REPO_ID   = f"{HF_USERNAME}/tourism-model"
TARGET_COL      = "ProdTaken"
MODEL_LOCAL     = "tourism_project/model_building/best_tourism_model.joblib"

def main():
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)
    train_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="train.csv",
                                 repo_type="dataset", token=hf_token)
    test_path  = hf_hub_download(repo_id=DATASET_REPO_ID, filename="test.csv",
                                 repo_type="dataset", token=hf_token)
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL]); y_train = train_df[TARGET_COL]
    X_test  = test_df.drop(columns=[TARGET_COL]);  y_test  = test_df[TARGET_COL]

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols     = X_train.select_dtypes(include=["object"]).columns.tolist()

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                        random_state=42, n_jobs=-1, tree_method="hist")
    pipe = Pipeline([("preprocessor", pre), ("classifier", clf)])

    neg, pos = (y_train==0).sum(), (y_train==1).sum()
    spw = neg / pos
    print(f"Class balance neg={neg} pos={pos} scale_pos_weight={spw:.2f}")

    param_grid = {
        "classifier__n_estimators":     [200, 400],
        "classifier__max_depth":        [4, 6],
        "classifier__learning_rate":    [0.05, 0.1],
        "classifier__subsample":        [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
        "classifier__scale_pos_weight": [1, spw],
    }

    mlflow.set_experiment("tourism-package-prediction")
    with mlflow.start_run(run_name="xgb_gridsearch"):
        gs = GridSearchCV(pipe, param_grid, scoring="f1", cv=5,
                          n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        for k, v in gs.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("cv_best_f1", gs.best_score_)
        print("Best params:", gs.best_params_)
        print("CV F1:", gs.best_score_)

        y_pred  = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = {
            "test_accuracy":  accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall":    recall_score(y_test, y_pred),
            "test_f1":        f1_score(y_test, y_pred),
            "test_roc_auc":   roc_auc_score(y_test, y_proba),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v); print(f"{k}: {v:.4f}")
        print("\nClassification report:\n", classification_report(y_test, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

        os.makedirs(os.path.dirname(MODEL_LOCAL), exist_ok=True)
        joblib.dump(best, MODEL_LOCAL)
        mlflow.sklearn.log_model(best, artifact_path="model")

    api = HfApi(token=hf_token)
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model",
                exist_ok=True, token=hf_token)
    api.upload_file(path_or_fileobj=MODEL_LOCAL,
                    path_in_repo="best_tourism_model.joblib",
                    repo_id=MODEL_REPO_ID, repo_type="model")
    print(f"Uploaded -> {MODEL_REPO_ID}/best_tourism_model.joblib")

if __name__ == "__main__":
    main()
