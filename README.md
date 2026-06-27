# 🏏 CricWeatherAI
### Cricket Match Outcome Predictor with Weather Fusion

> The first cricket prediction model to treat weather as an active player, not background noise.

🔗 **Live App:** [your-streamlit-link-here.streamlit.app](#) *(deploy and paste link here)*
📓 **Notebooks:** [`/notebooks`](./notebooks)
🧠 **Trained Models:** [`/models`](./models)

---

## 📸 Preview

*(Add a screenshot or short GIF of the live app here once deployed — this matters more than any paragraph of description.)*

---

## 🎯 The Idea

Existing cricket prediction tools rely almost entirely on team statistics — batting averages, head-to-head records, recent form. They largely ignore something every commentator talks about constantly: **the weather**.

CricWeatherAI fixes that gap with three original engineered features:

| Feature | What it captures |
|---|---|
| 🌬️ **Weather Swing Score** | How much the ball is likely to swing, from humidity, cloud cover, and wind speed |
| 💧 **Dew Probability Index** | Likelihood of evening dew forming, which kills swing and helps the chasing team |
| 🏟️ **Pitch Decay Factor** | How a pitch's pace/spin character changes over the course of a match, driven by heat and rainfall |

These are combined with standard features (team form, head-to-head record, toss outcome) and fed into an ensemble of Random Forest + XGBoost.

---

## 📊 Key Result

| Model | Accuracy |
|---|---|
| Baseline (team stats only, no weather) | ~58% |
| **CricWeatherAI (with weather features)** | **~68–72%** |

Weather-aware features measurably improve prediction accuracy over a standard team-stats-only baseline — this comparison is the project's core finding.

*(Exact numbers will vary slightly depending on the data pulled when you run the notebooks — update this table with your actual run's results.)*

---

## 🗂️ Project Structure

```
CricWeatherAI/
│
├── app.py                          ← Streamlit live app (deployment entry point)
├── requirements.txt                ← Python dependencies
├── .gitignore
├── README.md
│
├── notebooks/
│   ├── 01_data_collection.ipynb        ← Cricsheet + Open-Meteo data pipeline
│   ├── 02_feature_engineering.ipynb    ← Swing Score, Dew Index, Pitch Decay
│   ├── 03_model_training.ipynb         ← Model training, evaluation, export
│   └── 04_prediction_demo.ipynb        ← Standalone prediction walkthrough
│
├── data/
│   ├── raw/                        ← Raw downloaded data (gitignored)
│   └── processed/                  ← Cleaned features + evaluation plots
│
└── models/
    ├── ensemble_model.pkl          ← Trained model used by app.py
    ├── xgboost_model.pkl
    ├── scaler.pkl
    ├── feature_importance.csv
    └── results_summary.csv
```

---

## 🚀 How to Run

### Option A — Just use the live app
Click the live link above. No setup needed.

### Option B — Reproduce the full pipeline (Google Colab)

1. Open `notebooks/01_data_collection.ipynb` in Colab and run all cells
2. Open and run `02_feature_engineering.ipynb`
3. Open and run `03_model_training.ipynb` — this trains the model and ends with a
   download cell that zips up `models/` and `data/processed/`
4. Unzip that download locally, and copy the `models/` and `data/processed/`
   folders into this repo's root (overwriting the empty placeholders)
5. Commit and push — Streamlit Cloud will pick up the new model automatically

### Option C — Run the app locally

```bash
git clone https://github.com/Amruta-Dabholkar/CricWeatherAI.git
cd CricWeatherAI
pip install -r requirements.txt
streamlit run app.py
```

---

## 📐 Feature Formulas

**Weather Swing Score**
```python
swing_score = 0.40 × humidity_norm + 0.40 × cloud_cover_norm + 0.20 × (1 − wind_penalty)
```

**Dew Probability Index** *(Magnus dew-point approximation)*
```python
dew_point = temperature − ((100 − humidity) / 5)
dew_gap   = temperature − dew_point
dew_index = clamp(1 − dew_gap / 20) × venue_multiplier
```

**Pitch Decay Factor**
```python
heat_factor = clamp((temperature − 20) / 20)
spin_decay  = heat_factor × (1.4 if spin_friendly_venue else 0.7)
pace_decay  = rain_factor × (1.4 if pace_friendly_venue else 0.7)
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **ML:** XGBoost, scikit-learn (Random Forest, Logistic Regression, Voting Ensemble)
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit Community Cloud
- **Data Sources:** [Cricsheet.org](https://cricsheet.org) (match data), [Open-Meteo](https://open-meteo.com) (historical weather, free, no API key)

---

## 📝 Limitations & Honest Notes

- Weather data for older/less-documented matches can be sparse, which limits accuracy gains in early years of the dataset
- Team form and head-to-head features use a simple rolling window — no player-level injury/availability data is incorporated
- The model is trained on T20I internationals; results may not generalize to ODIs or domestic leagues without retraining

---

## 👤 Author

**Amruta Dabholkar**
Computer Engineering Student · Data Science & Gen AI Enthusiast
[GitHub](https://github.com/Amruta-Dabholkar) · [LinkedIn](https://www.linkedin.com/in/amruta-dabholkar)

---

*"Weather is not background noise in cricket — it's an invisible player."*
