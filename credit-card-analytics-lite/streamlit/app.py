# streamlit/app.py

# RUN:
# cd streamlit
# streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import joblib
import json

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Credit Card Analytics",
    page_icon="💳",
    layout="wide"
)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    return {

        'km': joblib.load("../models/kmeans_model.pkl"),

        'cs': joblib.load("../models/cluster_scaler.pkl"),

        'lm': joblib.load("../models/limit_predictor.pkl"),

        'ad': joblib.load("../models/anomaly_detector.pkl"),

        'as': joblib.load("../models/anomaly_scaler.pkl"),

        'lc': joblib.load("../models/le_city.pkl"),

        'lp': joblib.load("../models/le_payment.pkl"),

        'cluster_metrics': json.load(
            open("../models/cluster_metrics.json")
        ),

        'limit_metrics': json.load(
            open("../models/limit_metrics.json")
        ),

        'anomaly_metrics': json.load(
            open("../models/anomaly_metrics.json")
        )
    }

M = load_models()

persona_map = M['cluster_metrics']['persona_map']

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div style="
background:linear-gradient(135deg,#1E3A8A,#6D28D9);
padding:22px;
border-radius:15px;
margin-bottom:20px">

<h1 style="color:white;margin:0">
💳 Credit Card Analytics Engine
</h1>

<p style="color:#D8B4FE;margin-top:8px">
Customer Segmentation + Limit Prediction + Fraud Detection
</p>

</div>
""", unsafe_allow_html=True)

# =====================================================
# KPI CARDS
# =====================================================

c1, c2, c3 = st.columns(3)

c1.metric(
    "Silhouette Score",
    M['cluster_metrics']['silhouette_score']
)

c2.metric(
    "R² Score",
    M['limit_metrics']['r2']
)

c3.metric(
    "Anomaly Rate",
    f"{M['anomaly_metrics']['anomaly_rate']}%"
)

st.markdown("---")

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "👤 Persona",
    "💰 Credit Limit",
    "🚨 Anomaly Detection"
])

# =====================================================
# TAB 1 — PERSONA
# =====================================================

with tab1:

    st.subheader("Customer Persona Prediction")

    col1, col2 = st.columns(2)

    with col1:

        monthly_spend = st.number_input(
            "Monthly Spend",
            0,
            500000,
            75000,
            5000,
            key="p1"
        )

        travel_spend = st.number_input(
            "Travel Spend",
            0,
            200000,
            20000,
            1000,
            key="p2"
        )

        shopping_spend = st.number_input(
            "Shopping Spend",
            0,
            200000,
            25000,
            1000,
            key="p3"
        )

        food_spend = st.number_input(
            "Food Spend",
            0,
            100000,
            12000,
            1000,
            key="p4"
        )

    with col2:

        txn_count = st.slider(
            "Transaction Count",
            1,
            60,
            22,
            key="p5"
        )

        util_rate = st.slider(
            "Utilization Rate",
            0.01,
            0.99,
            0.52,
            0.01,
            key="p6"
        )

        reward_pts = st.number_input(
            "Reward Points",
            0,
            5000,
            800,
            key="p7"
        )

        intl_pct = st.slider(
            "International Txn %",
            0.0,
            0.50,
            0.08,
            0.01,
            key="p8"
        )

    if st.button(
        "Predict Persona",
        type="primary",
        use_container_width=True
    ):

        X = pd.DataFrame([{

            'monthly_spend': monthly_spend,

            'travel_spend': travel_spend,

            'shopping_spend': shopping_spend,

            'food_spend': food_spend,

            'txn_count': txn_count,

            'avg_txn_value':
                monthly_spend / (txn_count + 1),

            'util_rate': util_rate,

            'reward_pts_mo': reward_pts,

            'intl_txn_pct': intl_pct
        }])

        X_scaled = M['cs'].transform(X)

        cluster_id = int(
            M['km'].predict(X_scaled)[0]
        )

        persona = persona_map.get(
            str(cluster_id),
            "Unknown"
        )

        st.success(
            f"Predicted Persona: {persona}"
        )

        st.info(
            f"Cluster ID: {cluster_id}"
        )

# =====================================================
# TAB 2 — CREDIT LIMIT
# =====================================================

with tab2:

    st.subheader("Credit Limit Recommendation")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            22,
            65,
            32,
            key="c1"
        )

        income = st.number_input(
            "Monthly Income",
            10000,
            1000000,
            120000,
            10000,
            key="c2"
        )

        cibil = st.slider(
            "CIBIL Score",
            300,
            900,
            720,
            key="c3"
        )

        tenure = st.slider(
            "Card Tenure",
            1,
            84,
            24,
            key="c4"
        )

    with col2:

        city = st.selectbox(
            "City Tier",
            ["Tier1", "Tier2", "Tier3"],
            key="c5"
        )

        payment = st.selectbox(
            "Payment Behavior",
            ["Full", "Partial", "Minimum", "None"],
            key="c6"
        )

        spend = st.number_input(
            "Monthly Spend",
            0,
            500000,
            75000,
            5000,
            key="c7"
        )

        util = st.slider(
            "Utilization",
            0.01,
            0.99,
            0.52,
            0.01,
            key="c8"
        )

    if st.button(
        "Predict Credit Limit",
        type="primary",
        use_container_width=True
    ):

        city_enc = int(
            M['lc'].transform([city])[0]
        )

        pay_enc = int(
            M['lp'].transform([payment])[0]
        )

        txn_val = spend / 22

        X = pd.DataFrame([{

            'age': age,

            'income_mo': income,

            'cibil': cibil,

            'tenure_mo': tenure,

            'city_enc': city_enc,

            'monthly_spend': spend,

            'txn_count': 22,

            'avg_txn_value': txn_val,

            'util_rate': util,

            'payment_enc': pay_enc,

            'reward_pts_mo': 800,

            'intl_txn_pct': 0.08,

            'missed_payments': 0,

            'autopay_enabled': 1,

            'persona_enc': 0,

            'income_spend_ratio':
                spend / (income + 1),

            'spend_per_txn': txn_val,

            'high_util_flag':
                int(util > 0.70)

        }])[M['limit_metrics']['features']]

        pred = float(
            M['lm'].predict(X)[0]
        )

        pred = round(pred / 1000) * 1000

        st.metric(
            "Recommended Credit Limit",
            f"Rs {pred:,.0f}"
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={
                'text':
                "Recommended Credit Limit"
            }
        ))

        fig.update_layout(height=300)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =====================================================
# TAB 3 — ANOMALY
# =====================================================

with tab3:

    st.subheader("Fraud / Anomaly Detection")

    col1, col2 = st.columns(2)

    with col1:

        a_spend = st.number_input(
            "Monthly Spend ",
            0,
            1000000,
            90000,
            5000,
            key="a1"
        )

        a_txn = st.slider(
            "Txn Count ",
            1,
            100,
            35,
            key="a2"
        )

        a_util = st.slider(
            "Utilization Rate ",
            0.01,
            0.99,
            0.82,
            0.01,
            key="a3"
        )

    with col2:

        a_intl = st.slider(
            "International %",
            0.0,
            0.50,
            0.20,
            0.01,
            key="a4"
        )

        a_travel = st.number_input(
            "Travel Spend ",
            0,
            500000,
            30000,
            1000,
            key="a5"
        )

        a_rewards = st.number_input(
            "Reward Points ",
            0,
            5000,
            1200,
            key="a6"
        )

    if st.button(
        "Check Anomaly",
        type="primary",
        use_container_width=True
    ):

        X = pd.DataFrame([{

            'monthly_spend': a_spend,

            'txn_count': a_txn,

            'avg_txn_value':
                a_spend / (a_txn + 1),

            'util_rate': a_util,

            'intl_txn_pct': a_intl,

            'travel_spend': a_travel,

            'reward_pts_mo': a_rewards
        }])

        X_scaled = M['as'].transform(X)

        score = float(
            -M['ad'].score_samples(X_scaled)[0]
        )

        flag = (
            M['ad'].predict(X_scaled)[0] == -1
        )

        st.metric(
            "Anomaly Score",
            round(score, 4)
        )

        if flag:

            st.error(
                "🚨 HIGH RISK ANOMALY DETECTED"
            )

        else:

            st.success(
                "✅ NORMAL CUSTOMER BEHAVIOR"
            )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Built using K-Means + XGBoost + Isolation Forest + Streamlit"
)