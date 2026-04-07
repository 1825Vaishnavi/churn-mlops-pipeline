import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_processed_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1":       f1_score(y_test, preds),
        "auc":      roc_auc_score(y_test, proba),
    }


def train_and_log(name, model, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        # Log hyperparameters
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(f"{name} — AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f}")
        return metrics["auc"], mlflow.active_run().info.run_id


if __name__ == "__main__":
    mlflow.set_experiment("churn-prediction")

    X_train, X_test, y_train, y_test = load_processed_data()

    # Define 5 model configurations
    experiments = [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=42),
            {"max_iter": 1000, "random_state": 42}
        ),
        (
            "RandomForest",
            RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            {"n_estimators": 100, "max_depth": 6, "random_state": 42}
        ),
        (
            "XGBoost",
            XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                          use_label_encoder=False, eval_metric="logloss", random_state=42),
            {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}
        ),
        (
            "XGBoost-Tuned",
            XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                          use_label_encoder=False, eval_metric="logloss", random_state=42),
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05}
        ),
        (
            "LightGBM",
            LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                           random_state=42, verbose=-1),
            {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
        ),
    ]

    # Train all models, track the best one
    best_auc, best_run_id, best_name = 0, None, None
    for name, model, params in experiments:
        auc, run_id = train_and_log(name, model, params, X_train, X_test, y_train, y_test)
        if auc > best_auc:
            best_auc, best_run_id, best_name = auc, run_id, name

    # Register best model in MLflow Model Registry
    logger.info(f"\n🏆 Best model: {best_name} with AUC={best_auc:.4f}")
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="churn-model")
    print(f"\n✅ Training complete! Best model: {best_name} (AUC={best_auc:.4f})")
    print("Run 'mlflow ui' to see all experiments!")