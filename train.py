"""RETAIN AI 2.0 - churn model training.

Dataset: IBM Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)
Run:     python train.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges is blank for brand-new customers (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)
    return df.drop(columns=["customerID"])


def main(path: str) -> None:
    df = load(path)
    X, y = df[NUMERIC + CATEGORICAL], df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    Xtr_t = pre.fit_transform(X_tr)
    Xte_t = pre.transform(X_te)
    feature_names = list(pre.get_feature_names_out())

    # ~26% churn -> handle imbalance
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="auc",
        random_state=42, n_jobs=-1,
    )
    model.fit(Xtr_t, y_tr, eval_set=[(Xte_t, y_te)], verbose=False)

    proba = model.predict_proba(Xte_t)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_te, proba),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
    }
    print(classification_report(y_te, pred, target_names=["Stayed", "Churned"]))
    print(json.dumps(metrics, indent=2))

    joblib.dump(
        {"preprocessor": pre, "model": model, "feature_names": feature_names,
         "numeric": NUMERIC, "categorical": CATEGORICAL},
        "churn_model.joblib",
    )
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved churn_model.joblib and metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)