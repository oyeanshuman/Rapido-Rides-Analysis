# 🛵 Rapido Rides Analytics Project

> **Author:** Anshuman chandra  
> **Dataset:** Rapido ride data (50,000 records, June–August 2024)  
> **Tech Stack:** Python · scikit-learn · Streamlit · Plotly · Pandas · Seaborn · Matplotlib

---

## 📌 Project Overview

This project performs end-to-end analysis of Rapido ride-sharing data, covering:

1. **Data Cleaning & Preprocessing** – deduplication, type casting, negative-value correction, smart null imputation.
2. **Feature Engineering** – temporal features (hour, day, week, is\_night), label encoding of service types.
3. **Machine Learning** – predicting ride charges using Linear Regression and Random Forest Regressor.
4. **Interactive Web App** – Streamlit app with live predictions, visual dashboards, and branded Rapido UI.
5. **Business Insights** – pricing recommendations, fraud/outlier detection, operational efficiency suggestions.

---

## 📂 Project Files

| File | Description |
|---|---|
| `rides_data.csv` | Original raw dataset (50,000 rows) |
| `Rapido_dataset_cleaned.csv` | Cleaned dataset (auto-generated on first app run) |
| `Anshumanchandra_Rapido_Project.py` | **Main file** – Streamlit app (cleaning + ML + dashboard, all-in-one) |
| `generate_report.py` | Script to generate the Word project report |
| `regenerate_charts.py` | Script to regenerate saved chart images |
| `report_charts/` | Directory of chart images used in the report |
| `model_linear_regression.pkl` | Trained Linear Regression model (auto-saved on first run) |
| `model_random_forest.pkl` | Trained Random Forest model (auto-saved on first run) |
| `label_encoder_services.pkl` | Fitted LabelEncoder for service types (auto-saved on first run) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
| `Anshumanchandra_RapidoProjectReport.docx` | Full project report (Word) |

---

## 📊 Dataset
| Kaggle Dataset link : (rides_data.csv) "https://www.kaggle.com/datasets/vishaldeoprasad/bangalore-rapido-ride-services-dataset"
| Attribute | Value |
|---|---|
| Source | Rapido ride records |
| Rows | 50,000 |
| Columns | 13 (`services`, `date`, `time`, `ride_status`, `source`, `destination`, `duration`, `ride_id`, `distance`, `ride_charge`, `misc_charge`, `total_fare`, `payment_method`) |
| Date range | 2024-06-17 to 2024-08-16 |
| Service types | bike, auto, cab economy, bike lite, parcel |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, feature engineering |
| `numpy` | Numerical operations, polynomial trend fitting |
| `scikit-learn` | ML models, train/test split, label encoding, evaluation metrics |
| `streamlit` | Web app UI, dashboard, and caching |
| `plotly` | Interactive charts (scatter, bar, pie, area, histogram, heatmap) |
| `matplotlib` / `seaborn` | Correlation heatmap (static) |
| `joblib` | Model serialisation / persistence |

---

## ⚙️ Setup & Run Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit app

```bash
streamlit run Anshumanchandra_Rapido_Project.py
```

The app opens in your browser at `http://localhost:8501`.

> **First run behaviour:**
> - If `Rapido_dataset_cleaned.csv` is missing, the app **automatically cleans** the raw dataset and saves it.
> - If model `.pkl` files are missing, the app **automatically trains** both models and saves them.
> - Subsequent runs skip both steps and load everything instantly via Streamlit's cache.


## 🧹 Data Cleaning Pipeline

The cleaning logic runs the following 10 steps:

| Step | Operation |
|---|---|
| 1 | Load raw CSV |
| 2 | Standardise column names (lowercase + underscores) |
| 3 | Remove duplicate rows |
| 4 | Cast numeric columns (`duration`, `distance`, `ride_charge`, `misc_charge`, `total_fare`) using `pd.to_numeric` |
| 5 | Convert all negative numeric values to positive (`.abs()`) |
| 6 | Fill missing `ride_id` with placeholder `"RD0000000000000000"` |
| 7 | Impute missing `ride_charge` using group mean by `(service, distance_bucket)` with service-only mean as fallback |
| 8 | Fill `misc_charge` → `0.0`, recompute `total_fare`, fill `payment_method` → `"Unknown"` for cancelled rides |
| 9 | Cast `services`, `ride_status`, `payment_method` to `category` dtype |
| 10 | Save cleaned CSV to `Rapido_dataset_cleaned.csv` |

---

## 🔧 Feature Engineering

After loading the cleaned dataset, the following features are derived:

| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23), extracted from the `time` column |
| `day_of_week` | Day of week (0 = Monday … 6 = Sunday) |
| `month` | Calendar month number |
| `week` | ISO week number |
| `is_cancelled` | Binary flag: 1 if ride was cancelled, else 0 |
| `is_night` | Binary flag: 1 if hour is 22–5 (inclusive), else 0 |
| `services_display` | Title-cased service name for display (e.g. `"bike lite"` → `"Bike Lite"`) |
| `service_encoded` | Label-encoded integer representation of `services` |

---

## 🤖 Machine Learning

### Features used for training

| Feature | Description |
|---|---|
| `service_encoded` | Label-encoded service type |
| `distance` | Trip distance in km |
| `duration` | Trip duration in minutes |
| `hour` | Hour of day (0–23) |
| `day_of_week` | Day of week (0 = Mon … 6 = Sun) |
| `is_night` | 1 if hour is 22–5, else 0 |

**Target:** `ride_charge`  
**Split:** 80% train / 20% test (`random_state=42`)

### Models

| Model | Parameters |
|---|---|
| Linear Regression | Default scikit-learn settings |
| Random Forest Regressor | `n_estimators=100`, `random_state=42`, `n_jobs=-1` |

### Model Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 261.15 | 215.75 | ~0.00 |
| Random Forest | 271.11 | 225.11 | ~−0.08 |

> **Note:** The dataset's ride charges appear to have been generated with high randomness relative to the input features, resulting in low R² scores. In real-world data with genuine cost relationships, both models — especially Random Forest — would perform significantly better.

### Random Forest Feature Importances

| Feature | Approximate Importance |
|---|---|
| `distance` | ~39% |
| `duration` | ~23% |
| `hour` | ~17% |
| `service_encoded` | ~9% |
| `day_of_week` | ~7% |
| `is_night` | ~5% |

---

## 📈 Dashboard Sections

| Page | Description |
|---|---|
| 🔮 Ride Charge Predictor | Live prediction form — select service, distance, duration, hour, day; get charge estimate from LR, RF, or both |
| 📊 Executive Overview | KPIs (total rides, completions, cancellations, avg charge, avg distance), service mix pie chart, avg charge by service bar, ride status bar |
| 🔑 Key Drivers | Distance vs charge scatter (with trend line), charge distribution box plot by service, duration vs charge scatter, correlation heatmap |
| 🚗 Ride Analysis | Daily ride volume line chart, rides by hour bar chart, service × day density heatmap, avg distance by service, weekly revenue area chart |
| ⚠️ Customer & Risk Analysis | IQR outlier detection, outlier scatter, fraud suspects (short distance + high charge), cancellation rate by service |
| 🤖 ML Model Comparison | Actual vs predicted scatter (both models), RF feature importance bar chart, grouped metrics bar chart, metrics table |
| 💡 Recommended Actions | Data-driven recommendations on pricing, customer retention, operational efficiency, and fraud mitigation |

---

## 💡 Key Insights

- **Bike rides dominate** (30%+ of total rides), followed by Auto and Cab Economy.
- **Parcel rides** carry the highest average charge — an underutilised, high-margin segment.
- **Cancellation rate** is ~10% across all service types; Auto and Cab Economy lead.
- **Night hours (10 PM – 6 AM)** show consistent demand but no fare premium — night-surge pricing could improve both revenue and driver supply.
- **Short-distance, high-charge rides** (~suspicious cluster) warrant automated fraud-review workflows.
- **Distance is the top predictor** of ride charge (RF importance ≈ 39%), followed by Duration (~23%) and Hour (~17%).
- **Charge distribution is near-uniform** across ₹50–₹1,000, indicating the current pricing model does not differentiate sufficiently by distance or service tier.

---

## 📄 License

For academic / internship use only.
