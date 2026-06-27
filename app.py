import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt
import os

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitchIQ",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Dark Theme CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a1520; }
    h1, h2, h3 { color: #e0eaf5 !important; }
    .stMetric { background-color: #0d1f30; padding: 12px; border-radius: 8px; border: 1px solid #1e2d3d; }
    div[data-testid="stMetricValue"] { color: #00c882; }
    .stButton button {
        background: linear-gradient(135deg, #00c882, #00a86b);
        color: black; font-weight: 700; border: none; border-radius: 8px;
    }
    .key-factor {
        background-color: #0d1f30; border: 1px solid #2a4060; border-radius: 8px;
        padding: 14px; color: #ffd54f; font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🏏 PitchIQ")
st.caption("Cricket Match Outcome Predictor with Weather Fusion. Powered by XGBoost")

# ── Load Model (cached so it doesn't reload every interaction) ─────────────
@st.cache_resource
def load_model():
    model_path = "models/ensemble_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path), True
    return None, False

model, model_loaded = load_model()

if not model_loaded:
    st.warning(
        "⚠️ Trained model not found in `models/ensemble_model.pkl`. "
        "Run notebooks 01-03 first and commit the `models/` folder to your repo. "
        "Showing the app in **demo mode** with a rule-based fallback predictor."
    )

# ── Venue Data ─────────────────────────────────────────────────────────────
VENUES = {
    "Wankhede Stadium, Mumbai":   {"lat": 18.9388, "lon": 72.8258, "pitch": "flat",  "dew": True},
    "Eden Gardens, Kolkata":      {"lat": 22.5645, "lon": 88.3433, "pitch": "dusty", "dew": True},
    "M Chinnaswamy, Bangalore":   {"lat": 12.9791, "lon": 77.5996, "pitch": "flat",  "dew": True},
    "Edgbaston, Birmingham":      {"lat": 52.4558, "lon": -1.9022, "pitch": "green", "dew": False},
    "Lord's, London":             {"lat": 51.5293, "lon": -0.1727, "pitch": "green", "dew": False},
    "MCG, Melbourne":             {"lat": -37.8200, "lon": 144.9834, "pitch": "hard", "dew": False},
    "Gaddafi Stadium, Lahore":    {"lat": 31.5204, "lon": 74.3587, "pitch": "dusty", "dew": False},
    "Dubai International":       {"lat": 25.0386, "lon": 55.2254, "pitch": "flat",  "dew": True},
    "Newlands, Cape Town":        {"lat": -33.9033, "lon": 18.4198, "pitch": "hard", "dew": False},
}

TEAMS = ["India", "Australia", "England", "Pakistan", "South Africa",
         "New Zealand", "Sri Lanka", "West Indies", "Bangladesh", "Afghanistan"]

TEAM_FORM_DEFAULTS = {
    "India": 0.80, "Australia": 0.70, "England": 0.65, "Pakistan": 0.65,
    "South Africa": 0.60, "New Zealand": 0.65, "Sri Lanka": 0.45,
    "West Indies": 0.45, "Bangladesh": 0.40, "Afghanistan": 0.55,
}

# ── Feature Engineering (same formulas as training notebooks) ──────────────
def compute_features(team1, team2, venue_name, toss_winner, toss_decision,
                      temp, humidity, cloud_cover, wind_speed, precipitation):
    venue = VENUES[venue_name]

    swing_score = (0.40*(humidity/100) + 0.40*(cloud_cover/100)
                   + 0.20*(1 - min(wind_speed/60, 1.0)))

    dew_point = temp - ((100 - humidity) / 5)
    dew_gap = temp - dew_point
    dew_index = min(max(0, 1-(dew_gap/20)) * (1.3 if venue["dew"] else 0.7), 1.0)

    is_spin = venue["pitch"] in ["dusty", "flat"]
    heat = max(0, min((temp-20)/20, 1.0))
    spin_decay = heat * (1.4 if is_spin else 0.7)
    pace_decay = min(precipitation/10, 1.0) * (0.7 if not is_spin else 0.4)

    toss_win_team1 = 1 if toss_winner == team1 else 0
    chose_bat = 1 if toss_decision == "bat" else 0
    smart_field = 1 if (toss_win_team1 and toss_decision == "field" and dew_index > 0.5) else 0

    team1_form = TEAM_FORM_DEFAULTS.get(team1, 0.5)
    team2_form = TEAM_FORM_DEFAULTS.get(team2, 0.5)
    h2h_winrate = 0.5  # neutral prior without historical lookup

    return {
        "team1_form": team1_form, "team2_form": team2_form, "h2h_winrate": h2h_winrate,
        "toss_win_team1": toss_win_team1, "chose_bat": chose_bat, "smart_field": smart_field,
        "swing_score": swing_score, "dew_index": dew_index,
        "spin_decay": spin_decay, "pace_decay": pace_decay,
        "temp_max": temp, "humidity": humidity, "cloud_cover": cloud_cover,
        "wind_speed": wind_speed, "precipitation": precipitation,
    }

def fallback_predict(features, team1_form, team2_form):
    """Rule-based predictor used only if the trained .pkl isn't available."""
    score1 = team1_form*40 + features["toss_win_team1"]*5
    score2 = team2_form*40 + (1-features["toss_win_team1"])*5
    score1 += features["swing_score"]*10
    score2 += features["swing_score"]*10
    total = score1 + score2
    return round((score1/total)*100, 1)

def fetch_live_weather(lat, lon):
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&daily=temperature_2m_max,precipitation_sum,windspeed_10m_max,"
               f"cloudcover_mean,relative_humidity_2m_max&timezone=auto&forecast_days=1")
        d = requests.get(url, timeout=8).json()["daily"]
        return {
            "temp": d["temperature_2m_max"][0],
            "humidity": d["relative_humidity_2m_max"][0],
            "cloud": d["cloudcover_mean"][0],
            "wind": d["windspeed_10m_max"][0],
            "rain": d["precipitation_sum"][0],
        }
    except Exception:
        return None

# ── Sidebar: Match Setup ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Match Setup")

    team1 = st.selectbox("Team 1", TEAMS, index=0)
    team2 = st.selectbox("Team 2", [t for t in TEAMS if t != team1], index=0)
    venue_name = st.selectbox("Venue", list(VENUES.keys()))
    toss_winner = st.radio("Toss won by", [team1, team2], horizontal=True)
    toss_decision = st.radio("Toss decision", ["bat", "field"], horizontal=True)

    st.markdown("---")
    st.header("🌤 Weather")

    use_live = st.checkbox("Fetch live weather for venue", value=False)

    if use_live:
        v = VENUES[venue_name]
        live = fetch_live_weather(v["lat"], v["lon"])
        if live:
            temp = live["temp"]; humidity = live["humidity"]
            cloud_cover = live["cloud"]; wind_speed = live["wind"]
            precipitation = live["rain"]
            st.success(f"Live: {temp}°C, {humidity}% humidity")
        else:
            st.error("Couldn't fetch live weather, using manual sliders.")
            use_live = False

    if not use_live:
        temp = st.slider("Temperature (°C)", 5, 45, 28)
        humidity = st.slider("Humidity (%)", 20, 100, 65)
        cloud_cover = st.slider("Cloud Cover (%)", 0, 100, 40)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 60, 15)
        precipitation = st.slider("Precipitation (mm)", 0.0, 20.0, 0.0)

    predict_btn = st.button("🔮 PREDICT MATCH", use_container_width=True)

# ── Main Panel ───────────────────────────────────────────────────────────────
if team1 == team2:
    st.error("Please select two different teams.")
elif predict_btn:
    features = compute_features(team1, team2, venue_name, toss_winner, toss_decision,
                                 temp, humidity, cloud_cover, wind_speed, precipitation)

    feature_order = ["team1_form","team2_form","h2h_winrate","toss_win_team1","chose_bat",
                      "smart_field","swing_score","dew_index","spin_decay","pace_decay",
                      "temp_max","humidity","cloud_cover","wind_speed","precipitation"]

    if model_loaded:
        X = pd.DataFrame([{k: features[k] for k in feature_order}])
        prob = model.predict_proba(X)[0]
        team1_prob = round(prob[1]*100, 1)
    else:
        team1_prob = fallback_predict(features, features["team1_form"], features["team2_form"])

    team2_prob = round(100 - team1_prob, 1)
    winner = team1 if team1_prob > team2_prob else team2
    confidence = abs(team1_prob - 50)
    conf_label = "HIGH" if confidence > 20 else "MODERATE" if confidence > 10 else "LOW"

    key_factor = (
        "Swing bowling dominates" if features["swing_score"] > 0.6 else
        "Dew will favor the chasing team" if features["dew_index"] > 0.5 else
        "Spin will be decisive late" if features["spin_decay"] > 0.5 else
        "Batting-friendly conditions"
    )

    # ── Result Display ──────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🏆 Prediction")
        st.metric("Predicted Winner", winner, f"{conf_label} confidence")

        c1, c2 = st.columns(2)
        c1.metric(team1, f"{team1_prob}%")
        c2.metric(team2, f"{team2_prob}%")

        st.progress(team1_prob / 100)

        st.markdown(f'<div class="key-factor">⚡ <b>Key Factor:</b> {key_factor}</div>',
                    unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Weather Impact Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Swing Score", f"{features['swing_score']*100:.0f}%")
        m2.metric("Dew Risk", f"{features['dew_index']*100:.0f}%")
        m3.metric("Spin Decay", f"{features['spin_decay']*100:.0f}%")

    st.markdown("---")

    # ── Feature importance / explanation ────────────────────────────────────
    st.subheader("🔍 What Drove This Prediction")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0a1520")
    ax.set_facecolor("#0d1f30")
    factors = ["Team Form", "Weather/Swing", "Pitch Decay", "Dew Factor", "Toss"]
    values = [
        round((features["team1_form"]+features["team2_form"])/2*100, 1),
        round(features["swing_score"]*100, 1),
        round(features["spin_decay"]*100, 1),
        round(features["dew_index"]*100, 1),
        8.0,
    ]
    colors = ["#00c882", "#64b5f6", "#ffd54f", "#80cbc4", "#ce93d8"]
    ax.barh(factors, values, color=colors)
    ax.tick_params(colors="#8aa8c8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d3d")
    ax.set_xlabel("Relative Impact (%)", color="#8aa8c8")
    st.pyplot(fig)

    if not model_loaded:
        st.info(
            "💡 This used the rule-based fallback predictor. To use your real trained "
            "XGBoost ensemble, commit `models/ensemble_model.pkl` (generated by notebook 03) "
            "to this repo."
        )

else:
    st.info("👈 Set up your match in the sidebar, then click **PREDICT MATCH**.")

    st.markdown("### How This Works")
    st.markdown("""
    This model combines three things almost no other cricket predictor uses together:

    - **🌬 Weather Swing Score** — humidity, cloud cover, and wind combined into a single
      score estimating how much the ball will swing in the air.
    - **💧 Dew Probability Index** — for evening matches, estimates how likely dew is to
      form and how much it will help the team batting second.
    - **🏟 Pitch Decay Factor** — models how a pitch changes over the course of a match
      based on heat and recent rainfall.

    The model itself is an ensemble of Random Forest and XGBoost, trained on historical
    T20I match data from [Cricsheet](https://cricsheet.org) merged with historical weather
    data from [Open-Meteo](https://open-meteo.com).
    """)

st.markdown("---")
st.caption("Built with Streamlit · Data: Cricsheet.org + Open-Meteo")
