import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")

    # Drop customerID — not useful for prediction
    df = df.drop(columns=["customerID"])

    # TotalCharges has spaces instead of NaN — fix that
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Convert target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    logger.info(f"Churn rate: {df['Churn'].mean():.2%}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding categorical features...")

    # Binary yes/no columns
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

    # Label encode remaining categorical columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def save_processed(X_train, X_test, y_train, y_test, out_dir: str = "data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(f"{out_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{out_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)
    logger.info(f"Saved processed data to {out_dir}/")


if __name__ == "__main__":
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    df = clean_data(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_processed(X_train, X_test, y_train, y_test)
    print("✅ Data processing complete!")