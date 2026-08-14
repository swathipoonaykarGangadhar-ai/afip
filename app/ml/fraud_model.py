"""
Transaction-level fraud scoring model.
Trains an XGBoost classifier on engineered features from synthetic
transaction data. This is the "Fraud Detection Engine" from the
architecture doc — the thing that generates the alert that kicks
off the agentic investigation.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.joblib")


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].apply(lambda h: 1 if h < 6 or h > 22 else 0)

    # velocity: transactions per account in the dataset (proxy for real windowed velocity)
    velocity = df.groupby("account_id")["transaction_id"].transform("count")
    df["account_velocity"] = velocity

    # merchant risk: wire transfers and unknown locations are riskier
    df["is_wire"] = (df["merchant"] == "WIRE_TRANSFER").astype(int)
    df["is_unknown_location"] = (df["location"] == "Unknown").astype(int)

    df["amount_log"] = np.log1p(df["amount"])

    features = ["amount", "amount_log", "hour", "is_night",
                "account_velocity", "is_wire", "is_unknown_location"]
    return df[features]


def train(transactions: list[dict]):
    df = pd.DataFrame(transactions)
    X = featurize(df)
    y = df["label_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    report = classification_report(y_test, (preds > 0.5).astype(int))

    joblib.dump(model, MODEL_PATH)
    return {"auc": auc, "report": report, "model_path": MODEL_PATH}


def load_model():
    return joblib.load(MODEL_PATH)


def score_transaction(model, transaction: dict, account_velocity: int = 1) -> float:
    txn = dict(transaction)
    txn.setdefault("account_id", "UNKNOWN")
    df = pd.DataFrame([txn])
    df["transaction_id"] = df.get("transaction_id", "TX_SCORE")
    X = featurize(df)
    # override velocity with the caller-provided real value if given
    X["account_velocity"] = account_velocity
    return float(model.predict_proba(X)[0, 1])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.core.synthetic_data import generate_dataset

    ds = generate_dataset(n_customers=300, n_transactions=4000)
    result = train(ds["transactions"])
    print(f"AUC: {result['auc']:.4f}")
    print(result["report"])
