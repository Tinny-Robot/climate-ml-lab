# Climate ML Lab — Short-Term Weather Forecasting for Abeokuta, Nigeria

A machine learning project for next-day temperature forecasting in Abeokuta, Ogun State, Nigeria (7.16°N, 3.35°E). The project covers the full pipeline from data collection to a deployed Flask web dashboard.

---

## Model Performance

| Model | MAE  | RMSE  | R² |
|---|---|---|---|
| **Gradient Boosting** (best) | **0.48** | **0.60** | **0.90** |
| Lasso / Ridge | 0.47–0.48 | 0.59 | 0.91 |
| Random Forest | 0.49 | 0.61 | 0.90 |
| SVR | 0.53 | 0.67 | 0.88 |
| Decision Tree | 0.52 | 0.67 | 0.88 |

**Persistence baseline**: MAE = 0.47°C, R² = 0.90

---

## Quick Start

**Requirements**: Python 3.8+

```bash
# Install dependencies
pip install -r requirements.txt

# Start the dashboard
python app.py
# Open http://localhost:5000
```

---

## Project Structure

```
climate-ml-lab/
├── app.py                          # Flask web application
├── config.yaml                     # Project configuration
├── requirements.txt
│
├── notebooks/
│   ├── 01_data_collection.ipynb    # Fetch data from Open-Meteo API
│   ├── 01.2_data_dictionary.ipynb  # Variable descriptions
│   ├── 02_eda_analysis.ipynb       # Exploratory data analysis
│   ├── 03_model_training.ipynb     # Train & tune multiple models
│   ├── 04_model_evaluation.ipynb   # Model comparison and selection
│   └── 05_visualizations.ipynb     # Additional plots
│
├── src/
│   ├── data_loader.py              # API data fetching utilities
│   ├── preprocess.py               # Data cleaning & feature engineering
│   ├── train_models.py             # Model training & evaluation
│   ├── visualize.py                # Plotting utilities
│   └── utils.py                    # Shared helpers & paths
│
├── data/
│   ├── raw/ogun_weather.csv        # Original API data
│   └── processed/clean_weather.csv # Cleaned, engineered features
│
├── models/
│   └── saved_models/               # Serialised trained models (.joblib)
│
├── reports/
│   ├── evaluation_metrics.json     # Per-model test metrics
│   └── model_training_metrics.json # Full training run results
│
├── samp/                           # Forecast data files
├── visuals/                        # Generated charts (PNG)
├── templates/                      # Flask HTML templates
└── docs/                           # Technical documentation
```

---

## Data

- **Source**: Open-Meteo ERA5 Reanalysis API
- **Location**: Abeokuta, Ogun State, Nigeria
- **Period**: 2015-01-01 to present (~4,000+ daily records)
- **Variables**: Temperature, Humidity, Wind Speed, Precipitation, Surface Pressure
- **License**: CC-BY-4.0

---

## Methodology Summary

1. **Data collection** — Daily aggregates from Open-Meteo ERA5 hourly data.
2. **Preprocessing** — Time-series interpolation, lag-1 features, calendar features (month, day, weekday).
3. **Feature set** — 15 features; selected by correlation and feature importance.
4. **Training** — Temporal train/test split (80/20), TimeSeriesSplit cross-validation (5 folds).
5. **Tuning** — Grid search over Gradient Boosting, Random Forest, and SVR.
6. **Best model** — Gradient Boosting (learning_rate=0.05, max_depth=3, n_estimators=200).

Full methodology: [docs/METHODOLOGY.md](docs/METHODOLOGY.md)

---

## Dashboard

Running `python app.py` starts a Flask server at `http://localhost:5000` with three pages:

| Route | Description |
|---|---|
| `/dashboard` | Summary statistics, data quality, correlations, model performance charts |
| `/predict` | Select a date range to view forecast data |
| `/export` | Download forecast data as an Excel file |

---

## Troubleshooting

**Missing module**: `pip install -r requirements.txt`

**Port 5000 in use**: Edit the last line of `app.py` to use a different port.

**Dashboard shows no model metrics**: Ensure `reports/evaluation_metrics.json` exists (included in repo).

---

## Contact

**Author**: Nathaniel (Treasure) Handan  
**Email**: handanfoun@gmail.com  
**GitHub**: [@Tinny-Robot](https://github.com/Tinny-Robot)
