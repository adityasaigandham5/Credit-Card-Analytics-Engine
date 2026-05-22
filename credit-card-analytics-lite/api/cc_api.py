# api/cc_api.py

# RUN:
# cd api
# uvicorn cc_api:app --reload --port 8009

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np

import joblib
import json
import time

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(

    title="Credit Card Analytics API",

    version="1.0"
)

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"]
)

# ==========================================
# LOAD MODELS
# ==========================================

kmeans = joblib.load(
    "../models/kmeans_model.pkl"
)

cluster_scaler = joblib.load(
    "../models/cluster_scaler.pkl"
)

limit_model = joblib.load(
    "../models/limit_predictor.pkl"
)

anomaly_model = joblib.load(
    "../models/anomaly_detector.pkl"
)

anomaly_scaler = joblib.load(
    "../models/anomaly_scaler.pkl"
)

le_city = joblib.load(
    "../models/le_city.pkl"
)

le_payment = joblib.load(
    "../models/le_payment.pkl"
)

le_persona = joblib.load(
    "../models/le_persona.pkl"
)

# ==========================================
# LOAD METRICS
# ==========================================

cluster_metrics = json.load(
    open("../models/cluster_metrics.json")
)

limit_metrics = json.load(
    open("../models/limit_metrics.json")
)

anomaly_metrics = json.load(
    open("../models/anomaly_metrics.json")
)

persona_map = cluster_metrics[
    "persona_map"
]

# ==========================================
# FEATURE LISTS
# ==========================================

CLUSTER_FEATURES = cluster_metrics[
    "cluster_features"
]

LIMIT_FEATURES = limit_metrics[
    "features"
]

ANOMALY_FEATURES = anomaly_metrics[
    "anomaly_features"
]

# ==========================================
# INPUT SCHEMA
# ==========================================

class CustomerProfile(BaseModel):

    age: int = 32

    income_mo: float = 120000

    cibil: int = 720

    tenure_mo: int = 24

    city_tier: str = "Tier1"

    monthly_spend: float = 75000

    food_spend: float = 15000

    travel_spend: float = 20000

    shopping_spend: float = 25000

    txn_count: int = 22

    avg_txn_value: float = 3400

    util_rate: float = 0.52

    payment_behavior: str = "Full"

    reward_pts_mo: int = 800

    intl_txn_pct: float = 0.08

    missed_payments: int = 0

    autopay_enabled: int = 1

# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {

        "status": "running",

        "models": [

            "kmeans",

            "xgboost",

            "isolation_forest"
        ]
    }

# ==========================================
# PERSONA PREDICTION
# ==========================================

@app.post("/predict/persona")
def predict_persona(req: CustomerProfile):

    start = time.time()

    feats = pd.DataFrame([{

        'monthly_spend': req.monthly_spend,

        'travel_spend': req.travel_spend,

        'shopping_spend': req.shopping_spend,

        'food_spend': req.food_spend,

        'txn_count': req.txn_count,

        'avg_txn_value': req.avg_txn_value,

        'util_rate': req.util_rate,

        'reward_pts_mo': req.reward_pts_mo,

        'intl_txn_pct': req.intl_txn_pct
    }])

    X_scaled = cluster_scaler.transform(
        feats
    )

    cluster_id = int(
        kmeans.predict(X_scaled)[0]
    )

    persona = persona_map.get(
        str(cluster_id),
        "Unknown"
    )

    return {

        "cluster_id": cluster_id,

        "persona": persona,

        "latency_ms": round(
            (time.time() - start) * 1000,
            2
        )
    }

# ==========================================
# CREDIT LIMIT PREDICTION
# ==========================================

@app.post("/predict/limit")
def predict_limit(req: CustomerProfile):

    start = time.time()

    city_enc = int(
        le_city.transform([req.city_tier])[0]
    )

    payment_enc = int(
        le_payment.transform(
            [req.payment_behavior]
        )[0]
    )

    persona_enc = 0

    income_spend_ratio = (
        req.monthly_spend /
        (req.income_mo + 1)
    )

    spend_per_txn = (
        req.monthly_spend /
        (req.txn_count + 1)
    )

    high_util_flag = int(
        req.util_rate > 0.70
    )

    X = pd.DataFrame([{

        'age': req.age,

        'income_mo': req.income_mo,

        'cibil': req.cibil,

        'tenure_mo': req.tenure_mo,

        'city_enc': city_enc,

        'monthly_spend': req.monthly_spend,

        'txn_count': req.txn_count,

        'avg_txn_value': req.avg_txn_value,

        'util_rate': req.util_rate,

        'payment_enc': payment_enc,

        'reward_pts_mo': req.reward_pts_mo,

        'intl_txn_pct': req.intl_txn_pct,

        'missed_payments': req.missed_payments,

        'autopay_enabled': req.autopay_enabled,

        'persona_enc': persona_enc,

        'income_spend_ratio':
            income_spend_ratio,

        'spend_per_txn':
            spend_per_txn,

        'high_util_flag':
            high_util_flag

    }])[LIMIT_FEATURES]

    pred = float(
        limit_model.predict(X)[0]
    )

    pred = round(pred / 1000) * 1000

    return {

        "recommended_limit":
            int(pred),

        "r2_score":
            limit_metrics['r2'],

        "latency_ms": round(
            (time.time() - start) * 1000,
            2
        )
    }

# ==========================================
# ANOMALY DETECTION
# ==========================================

@app.post("/predict/anomaly")
def detect_anomaly(req: CustomerProfile):

    start = time.time()

    feats = pd.DataFrame([{

        'monthly_spend': req.monthly_spend,

        'txn_count': req.txn_count,

        'avg_txn_value': req.avg_txn_value,

        'util_rate': req.util_rate,

        'intl_txn_pct': req.intl_txn_pct,

        'travel_spend': req.travel_spend,

        'reward_pts_mo': req.reward_pts_mo
    }])

    X_scaled = anomaly_scaler.transform(
        feats
    )

    score = float(
        -anomaly_model.score_samples(
            X_scaled
        )[0]
    )

    is_anomaly = (
        anomaly_model.predict(
            X_scaled
        )[0] == -1
    )

    return {

        "is_anomaly":
            bool(is_anomaly),

        "anomaly_score":
            round(score, 4),

        "status":
            "INVESTIGATE"
            if is_anomaly
            else "NORMAL",

        "latency_ms": round(
            (time.time() - start) * 1000,
            2
        )
    }