"""RETAIN AI 2.0 - FastAPI backend with SHAP explanations.

Run: uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from typing import List

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

bundle = joblib.load("churn_model.joblib")
PRE, MODEL = bundle["preprocessor"], bundle["model"]
FEATURES = bundle["feature_names"]
COLS = bundle["numeric"] + bundle["categorical"]
EXPLAINER = shap.TreeExplainer(MODEL)

app = FastAPI(title="RETAIN AI 2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def tier(p: float) -> str:
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"


def score(customers: List[Customer], top_k: int = 5):
    df = pd.DataFrame([c.model_dump() for c in customers])[COLS]
    X = PRE.transform(df)
    proba = MODEL.predict_proba(X)[:, 1]
    shap_vals = EXPLAINER.shap_values(X)
    out = []
    for i, p in enumerate(proba):
        order = np.argsort(-np.abs(shap_vals[i]))[:top_k]
        drivers = [{
            "feature": FEATURES[j],
            "shap_value": round(float(shap_vals[i][j]), 4),
            "effect": "increases churn risk" if shap_vals[i][j] > 0
                      else "reduces churn risk",
        } for j in order]
        monthly = float(df.loc[i, "MonthlyCharges"])
        out.append({
            "churn_probability": round(float(p), 4),
            "risk_tier": tier(float(p)),
            # simple retention-priority score: expected annual revenue at risk
            "revenue_at_risk": round(float(p) * monthly * 12, 2),
            "top_drivers": drivers,
        })
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    return score([customer])[0]


@app.post("/predict/batch")
def predict_batch(customers: List[Customer]):
    results = score(customers)
    # rank by revenue at risk so the team can prioritise retention outreach
    return sorted(results, key=lambda r: r["revenue_at_risk"], reverse=True)


@app.get("/global-importance")
def global_importance():
    imp = MODEL.feature_importances_
    order = np.argsort(-imp)[:15]
    return [{"feature": FEATURES[i], "importance": round(float(imp[i]), 4)}
            for i in order]