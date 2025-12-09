import os
import io
import pickle
from datetime import date, time, datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crime Case Closure Predictor", page_icon="🔒", layout="centered")

# -----------------------------
# Helpers
# -----------------------------
def to_age_group(v):
    try:
        v = int(v)
    except Exception:
        return 2
    if v < 18: return 0
    if v < 30: return 1
    if v < 50: return 2
    if v < 70: return 3
    return 4

def safe_transform(le, value):
    """Transform a single value with LabelEncoder; if unseen, map to 'Other' or the first class."""
    val = str(value)
    classes = list(le.classes_)
    if val in classes:
        return int(le.transform([val])[0])
    if "Other" in classes:
        return int(le.transform(["Other"])[0])
    # fallback: first class
    return int(le.transform([classes[0]])[0])

def enc_transform_series(le, series):
    """Transform a Series with LabelEncoder using safe_transform for unseen."""
    return series.astype(str).apply(lambda v: safe_transform(le, v))

def parse_hour_from_time_str(s, default=9):
    """Parse 'HH:MM' or datetime-like to hour; fallback to default."""
    try:
        # if already time
        if isinstance(s, (time,)):
            return s.hour
        # try pandas to_datetime
        ts = pd.to_datetime(s, errors="coerce")
        if pd.notna(ts):
            return int(getattr(ts, "hour", default))
        # regex fallback
        s = str(s)
        m = pd.Series([s]).str.extract(r"(\d{1,2}):(\d{2})")
        if pd.notna(m.iloc[0,0]):
            return int(m.iloc[0,0])
    except Exception:
        pass
    return default

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_model():
    # Try pickle then joblib
    try:
        with open("svm_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        import joblib
        return joblib.load("svm_model.pkl")

@st.cache_resource
def load_encoders():
    """Load label encoders if present: keys = city, crime, weapon, domain, gender."""
    try:
        with open("encoders.pkl", "rb") as f:
            enc = pickle.load(f)
        # quick sanity
        for k in ["city", "crime", "weapon", "domain", "gender"]:
            _ = enc[k].classes_
        return enc
    except Exception as e:
        st.warning("encoders.pkl not found or unreadable. Unseen categories will be mapped to a default.")
        return None

@st.cache_resource
def load_imputer():
    try:
        import joblib
        return joblib.load("imputer.pkl")
    except Exception:
        return None

model = load_model()
encoders = load_encoders()
imputer = load_imputer()

# Expected feature order: take from model first (best), else hardcode 14-feature list
EXPECTED_COLS = list(getattr(model, "feature_names_in_", []))
if not EXPECTED_COLS:
    EXPECTED_COLS = [
        "Victim Age",
        "report_hour",
        "report_dayofweek",
        "report_month",
        "report_year",
        "occurrence_hour",
        "days_to_report",
        "victim_age_group",
        "city_encoded",
        "crime_code_encoded",
        "weapon_encoded",
        "domain_encoded",
        "gender_encoded",
        "desc_word_count",
    ]

def align_and_clean_numeric(df):
    # Keep only expected columns, add missing as 0, and ensure numeric
    out = df.copy()
    for c in EXPECTED_COLS:
        if c not in out.columns:
            out[c] = 0
    out = out[EXPECTED_COLS]
    out = out.replace([np.inf, -np.inf], np.nan)
    # apply external imputer if available (KNN or SimpleImputer)
    if imputer is not None:
        arr = imputer.transform(out.values.astype(float))
        out = pd.DataFrame(arr, columns=EXPECTED_COLS, index=out.index)
    else:
        out = out.fillna(out.median(numeric_only=True))
    # final numeric cast
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out

def predict_df(X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return pred, proba
    pred = model.predict(X)
    return pred, None

# -----------------------------
# UI
# -----------------------------
st.title("🔒 Crime Case Closure Prediction")
st.caption(f"Model expects {len(EXPECTED_COLS)} features: {EXPECTED_COLS}")

tab_single, tab_batch, tab_about = st.tabs(["Single Prediction", "Batch Predictions", "About"])

# ========== Single ==========
with tab_single:
    with st.form("single_form", clear_on_submit=False):
        st.subheader("Enter case details (raw inputs)")

        c1, c2 = st.columns(2)
        with c1:
            date_reported = st.date_input("Date Reported", value=date(2020,1,1))
            time_reported = st.time_input("Time Reported", value=time(9,0))
            date_occ = st.date_input("Date of Occurrence", value=date(2020,1,1))
            time_occ = st.time_input("Time of Occurrence", value=time(12,0))
            victim_age = st.number_input("Victim Age", min_value=0, max_value=120, value=30)

        with c2:
            # Use encoder classes if available, else free text defaults
            if encoders:
                city = st.selectbox("City", encoders["city"].classes_.tolist())
                crime_code = st.selectbox("Crime Code", encoders["crime"].classes_.tolist())
                weapon_used = st.selectbox("Weapon Used", encoders["weapon"].classes_.tolist())
                crime_domain = st.selectbox("Crime Domain", encoders["domain"].classes_.tolist())
                victim_gender = st.selectbox("Victim Gender", encoders["gender"].classes_.tolist())
            else:
                city = st.text_input("City", "Mumbai")
                crime_code = st.text_input("Crime Code", "IPC-123")
                weapon_used = st.text_input("Weapon Used", "None")
                crime_domain = st.text_input("Crime Domain", "Urban")
                victim_gender = st.selectbox("Victim Gender", ["Male","Female","Other"])
            crime_desc = st.text_area("Crime Description (optional)", "")

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build engineered + encoded features
        report_hour = time_reported.hour
        report_dayofweek = date_reported.weekday()
        report_month = date_reported.month
        report_year = date_reported.year
        occurrence_hour = time_occ.hour
        days_to_report = (date_reported - date_occ).days
        victim_age_group = to_age_group(victim_age)
        desc_word_count = len(crime_desc.split()) if crime_desc.strip() else 0

        if encoders:
            city_encoded   = safe_transform(encoders["city"], city)
            crime_encoded  = safe_transform(encoders["crime"], crime_code)
            weapon_encoded = safe_transform(encoders["weapon"], weapon_used)
            domain_encoded = safe_transform(encoders["domain"], crime_domain)
            gender_encoded = safe_transform(encoders["gender"], victim_gender)
        else:
            # If encoders are missing, fall back to zeros (not ideal but avoids crash)
            city_encoded = crime_encoded = weapon_encoded = domain_encoded = gender_encoded = 0

        row = pd.DataFrame([{
            "Victim Age": int(victim_age),
            "report_hour": report_hour,
            "report_dayofweek": report_dayofweek,
            "report_month": report_month,
            "report_year": report_year,
            "occurrence_hour": occurrence_hour,
            "days_to_report": days_to_report,
            "victim_age_group": victim_age_group,
            "city_encoded": city_encoded,
            "crime_code_encoded": crime_encoded,
            "weapon_encoded": weapon_encoded,
            "domain_encoded": domain_encoded,
            "gender_encoded": gender_encoded,
            "desc_word_count": desc_word_count,
        }])

        X = align_and_clean_numeric(row)

        try:
            pred, proba = predict_df(X)
        except Exception as e:
            st.error("Prediction failed due to column mismatch.")
            st.write("Features sent to model:"); st.dataframe(X)
            st.write("Expected order:"); st.code("\n".join(EXPECTED_COLS))
            st.exception(e); st.stop()

        st.divider(); st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Prediction", "Closed" if int(pred[0])==1 else "Not Closed")
        with c2: st.metric("Closure Probability", f"{float(proba[0]):.1%}" if proba is not None else "N/A")
        with c3:
            risk = "Low Risk" if (proba is not None and float(proba[0])>=0.5) else "High Risk"
            st.metric("Risk Level", risk)

        with st.expander("View engineered features"):
            st.dataframe(X)

# ========== Batch ==========
with tab_batch:
    st.subheader("Batch Predictions (CSV/XLSX)")
    st.caption("Upload raw columns; the app will engineer/encode to the 14-feature set the model expects.")

    # Template for raw input
    template_raw_cols = [
        "Date Reported","Time Reported",
        "Date of Occurrence","Time of Occurrence",
        "Victim Age","Victim Gender","City","Crime Code","Weapon Used","Crime Domain",
        "Crime Description"
    ]
    example_raw = pd.DataFrame([{
        "Date Reported":"2020-01-01","Time Reported":"09:00",
        "Date of Occurrence":"2020-01-01","Time of Occurrence":"12:00",
        "Victim Age":30,"Victim Gender":"Male","City":"Mumbai","Crime Code":"IPC-123",
        "Weapon Used":"None","Crime Domain":"Urban","Crime Description":"phone theft near bus stop"
    }], columns=template_raw_cols)
    buf = io.StringIO(); example_raw.to_csv(buf, index=False)
    st.download_button("Download RAW template CSV", buf.getvalue(), "raw_template.csv", "text/csv")

    up = st.file_uploader("Upload RAW CSV or Excel", type=["csv","xlsx","xls"])
    if up is not None:
        try:
            df_in = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as e:
            st.error("Failed to read uploaded file."); st.exception(e); st.stop()

        st.write("Preview of uploaded raw data:"); st.dataframe(df_in.head())

        # Build features from raw
        # Parse dates/times
        dr = pd.to_datetime(df_in.get("Date Reported"), errors="coerce")
        dor = pd.to_datetime(df_in.get("Date of Occurrence"), errors="coerce")

        # Hours
        tr_series = df_in.get("Time Reported")
        if tr_series is None:
            report_hour = pd.Series([9]*len(df_in))
        else:
            report_hour = tr_series.apply(parse_hour_from_time_str)

        to_series = df_in.get("Time of Occurrence")
        occurrence_hour = to_series.apply(parse_hour_from_time_str) if to_series is not None else report_hour

        features_df = pd.DataFrame({
            "Victim Age": pd.to_numeric(df_in.get("Victim Age"), errors="coerce"),
            "report_hour": report_hour,
            "report_dayofweek": dr.dt.dayofweek,
            "report_month": dr.dt.month,
            "report_year": dr.dt.year,
            "occurrence_hour": occurrence_hour,
            "days_to_report": (dr - dor).dt.days,
            "victim_age_group": pd.cut(pd.to_numeric(df_in.get("Victim Age"), errors="coerce"),
                                       bins=[0,18,30,50,70,120], labels=False),
            "desc_word_count": df_in.get("Crime Description", "").astype(str).str.split().str.len(),
        })

        # Encoded categoricals
        if encoders:
            features_df["city_encoded"]        = enc_transform_series(encoders["city"],   df_in.get("City", ""))
            features_df["crime_code_encoded"]  = enc_transform_series(encoders["crime"],  df_in.get("Crime Code", ""))
            features_df["weapon_encoded"]      = enc_transform_series(encoders["weapon"], df_in.get("Weapon Used", ""))
            features_df["domain_encoded"]      = enc_transform_series(encoders["domain"], df_in.get("Crime Domain", ""))
            features_df["gender_encoded"]      = enc_transform_series(encoders["gender"], df_in.get("Victim Gender", ""))
        else:
            # no encoders: fill zeros
            for c in ["city_encoded","crime_code_encoded","weapon_encoded","domain_encoded","gender_encoded"]:
                features_df[c] = 0

        Xb = align_and_clean_numeric(features_df)

        try:
            pred_b, proba_b = predict_df(Xb)
        except Exception as e:
            st.error("Batch prediction failed due to column mismatch.")
            st.write("First row features:"); st.dataframe(Xb.head(1))
            st.write("Expected order:"); st.code("\n".join(EXPECTED_COLS))
            st.exception(e); st.stop()

        out = df_in.copy()
        out["prediction"] = pred_b.astype(int)
        if proba_b is not None: out["prob_closed"] = proba_b

        st.success("Batch predictions complete."); st.dataframe(out.head())
        out_buf = io.StringIO(); out.to_csv(out_buf, index=False)
        st.download_button("Download Results CSV", out_buf.getvalue(), "batch_results.csv", "text/csv")

# ========== About ==========
with tab_about:
    st.subheader("About this app")
    st.markdown(
        "- Model expects the following 14 features and encodings (LabelEncoder):\n"
        f"`{EXPECTED_COLS}`\n\n"
        "- Make sure `encoders.pkl` (with keys: city, crime, weapon, domain, gender) is present; "
        "unseen categories will be mapped to a default class."
    )