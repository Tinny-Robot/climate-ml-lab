# Reproduction Guide

## Requirements

- Python 3.8 – 3.11
- pip 21.0+

## Setup

```bash
git clone https://github.com/Tinny-Robot/climate-ml-lab.git
cd climate-ml-lab
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Notebooks

Open and run in order:

1. `notebooks/01_data_collection.ipynb` — fetch raw data
2. `notebooks/02_eda_analysis.ipynb` — exploratory analysis
3. `notebooks/03_model_training.ipynb` — train models, produces `reports/` and `models/`
4. `notebooks/04_model_evaluation.ipynb` — evaluate and compare
5. `notebooks/05_visualizations.ipynb` — additional plots saved to `visuals/`

All random seeds are fixed at 42. Re-running the notebooks produces identical results.

## Running the Dashboard

```bash
python app.py
# Visit http://localhost:5000
```

## Expected Outputs

| Path | Description |
|---|---|
| `data/processed/clean_weather.csv` | Preprocessed dataset |
| `models/saved_models/best_model_GradientBoosting.joblib` | Trained model |
| `reports/evaluation_metrics.json` | Per-model test metrics |
| `visuals/*.png` | All generated plots |
