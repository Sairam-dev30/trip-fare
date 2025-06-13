import streamlit as st
import numpy as np
import pandas as pd
import pickle
import pytz
from datetime import datetime

# ── Page config (first line) ──
st.set_page_config(page_title="NYC Taxi Fare Predictor", layout="centered")

# ── Custom CSS for black background & styling ──
st.markdown(
    """
    <style>
    .stApp, .reportview-container {
        background-color: #000000;
        color: #ffffff;
    }
    h1, h2, h3, .stSubheader, .stText {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
    }
    .stNumberInput>div>div>input {
        background-color: #222222;
        color: #ffffff;
    }
    .stDateInput>div>div>input, .stTimeInput>div>div>input {
        background-color: #222222;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── Header Image (smaller) ──
st.image(
    "https://i.imgur.com/2yaf2wb.png",
    width=300  # adjust width as needed
)

st.title("🚖 NYC Taxi Fare Predictor")
st.write("Enter trip details below to estimate your fare.", unsafe_allow_html=True)

# ── Load trained model ──
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ── Haversine function ──
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ── Input Form ──
with st.form("fare_form"):
    st.subheader("Trip Information")

    pickup_lat       = st.number_input("Pickup Latitude", format="%.6f", value=40.761433)
    pickup_lon       = st.number_input("Pickup Longitude", format="%.6f", value=-73.979816)
    dropoff_lat      = st.number_input("Dropoff Latitude", format="%.6f", value=40.651311)
    dropoff_lon      = st.number_input("Dropoff Longitude", format="%.6f", value=-73.880333)
    passenger_count  = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)

    pickup_date      = st.date_input("Pickup Date", value=datetime.today().date())
    pickup_time      = st.time_input("Pickup Time", value=datetime.now().time())

    submitted        = st.form_submit_button("Predict Fare")

# ── Prediction & Map ──
if submitted:
    try:
        # Combine date & time, convert to EDT
        dt = datetime.combine(pickup_date, pickup_time)
        eastern = pytz.timezone("US/Eastern")
        dt = eastern.localize(dt) if dt.tzinfo is None else dt

        hour       = dt.hour
        day_name   = dt.strftime("%A")
        is_night   = 1 if hour < 6 or hour > 22 else 0
        is_weekend = 1 if day_name in ["Saturday", "Sunday"] else 0
        am_pm_flag = 1 if hour >= 12 else 0

        # Compute distance & log-transform
        dist       = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        log_dist   = np.log1p(dist)

        # One-hot encode pickup_day (Monday dropped)
        days = ["Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_enc = {f"pickup_day_{d}": 1 if day_name == d else 0 for d in days}

        # Build feature dict
        features = {
            "passenger_count": passenger_count,
            "trip_distance":   log_dist,
            "pickup_hour":     hour,
            "is_night":        is_night,
            "is_weekend":      is_weekend,
            "am_pm_PM":        am_pm_flag,
            **day_enc
        }

        # Align with model features
        X_input = pd.DataFrame([features], columns=model.feature_names_in_).fillna(0)

        # Predict & invert log
        log_pred = model.predict(X_input)[0]
        fare = np.expm1(log_pred)

        st.success(f"💵 Estimated Fare: ${fare:.2f}")

        # Display map of pickup & dropoff
        st.map(pd.DataFrame({
            "lat": [pickup_lat, dropoff_lat],
            "lon": [pickup_lon, dropoff_lon]
        }))

    except Exception as e:
        st.error(f"❗ Invalid input: {e}")
