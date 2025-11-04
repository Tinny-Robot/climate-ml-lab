# 🌦️ Climate-ML-Lab

**Climate-ML-Lab** is a machine learning research project focused on short-term climate forecasting for Ogun State, Nigeria — specifically the Abeokuta region.  
The project leverages data from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) and applies multiple ML modeling techniques implemented in Python to predict short-term temperature variations.

---

## 📊 Project Overview

This system:
- Collects and preprocesses daily weather data (temperature, humidity, wind speed, rainfall, pressure)
- Applies several ML models (Random Forest, Gradient Boosting, SVR, etc.)
- Evaluates models based on MAE, RMSE, and R²
- Selects the best-performing model for deployment

---

## 🧠 Methodology

1. **Data Collection** – Fetch weather data from Open-Meteo for Abeokuta station (7.16°N, 3.35°E).  
2. **Preprocessing** – Clean data, handle missing values, create lag features, normalize features.  
3. **Modeling** – Train multiple ML models and tune hyperparameters.  
4. **Evaluation** – Compare model performance on test data.  
5. **Forecasting** – Generate next-day or short-term forecasts.  

---

## ⚙️ Setup

```bash
# Clone repo
git clone https://github.com/<your-username>/climate-ml-lab.git
cd climate-ml-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (on Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
````

---

## 📂 Data Source

* **Provider:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
* **Location:** Abeokuta, Ogun State, Nigeria
* **Variables:** Temperature, Humidity, Wind Speed, Precipitation, Pressure
* **Format:** CSV

---

## 👨‍💻 Author

**Nathaniel (Treasure) Handan**
Community Lead, AI Bauchi | AWS Community Builder | Mechatronics Engineer
📧 [handanfoun@gmail.com](mailto:handanfoun@gmail.com)

---

## 🔒 License

This repository is **private** and intended for academic and research purposes only.
