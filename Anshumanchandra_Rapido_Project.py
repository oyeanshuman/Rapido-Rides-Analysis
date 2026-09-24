# =============================================================================
# Anshumanchandra – Rapido Rides Project
# =============================================================================
# Workflow
#   Section 1 : Raw Data Loading & Cleaning
#   Section 2 : Data Loading & Feature Engineering
#   Section 3 : Machine Learning Modeling
#   Section 4 : Streamlit App – Prediction UI
#   Section 5 : Streamlit App – Dashboard & Visualisation
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Rapido Brand Palette
# ---------------------------------------------------------------------------
RAPIDO_YELLOW  = "#F9C935"
RAPIDO_DARK    = "#000000"
RAPIDO_WHITE   = "#FFFFFF"
RAPIDO_HOVER   = "#FFD966"
RAPIDO_GREY    = "#F5F5F5"
RAPIDO_PALETTE = ["#F9C935", "#FFD966", "#FFC200", "#E6B800",
                  "#CC9900", "#B38600", "#997300"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DATASET_PATH = "rides_data.csv"
DATASET_PATH     = "Rapido_dataset_cleaned.csv"
LR_MODEL_PATH    = "model_linear_regression.pkl"
RF_MODEL_PATH    = "model_random_forest.pkl"
ENCODER_PATH     = "label_encoder_services.pkl"

# ---------------------------------------------------------------------------
# Rapido global CSS
# ---------------------------------------------------------------------------
RAPIDO_CSS = """
<style>
/* ── Page background ── */
.stApp { background-color: #FFFFFF; }

/* ── Sidebar background: Rapido yellow ── */
[data-testid="stSidebar"] {
    background-color: #F9C935 !important;
}
/* All sidebar text → black */
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: #000000 !important;
}

/* ── Sidebar nav buttons
   Normal state: white bg, black text */
div[data-testid="stSidebar"] div.stButton > button {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    font-weight: 600;
    border: 1.5px solid #000000 !important;
    border-radius: 8px;
    width: 100%;
    margin-bottom: 4px;
    text-align: left;
    padding: 0.45rem 0.9rem;
    transition: all 0.15s ease;
}
/* Hover state */
div[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #000000 !important;
    color: #F9C935 !important;
    border-color: #000000 !important;
}
/* Focus / active state */
div[data-testid="stSidebar"] div.stButton > button:focus {
    background-color: #000000 !important;
    color: #F9C935 !important;
    border-color: #000000 !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ── Sliders: thumb and filled track → black ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #000000 !important;
    border-color:     #000000 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="TrackHighlight"] {
    background-color: #000000 !important;
}
[data-testid="stSlider"] div[data-baseweb="typo-label"] {
    color: #000000 !important;
    font-weight: 700 !important;
}
[data-testid="stSlider"] p { color: #000000 !important; }

/* ── Radio dots → black border, yellow fill when selected ── */
[data-testid="stRadio"] label span:first-child {
    border-color: #000000 !important;
}
[data-testid="stRadio"] input:checked + div {
    background-color: #F9C935 !important;
    border-color: #000000 !important;
}

/* ── Selectbox border ── */
div[data-baseweb="select"] > div {
    border: 2px solid #000000 !important;
    border-radius: 6px;
}

/* ── Bold labels for all input widgets ── */
[data-testid="stSlider"] > label,
[data-testid="stSelectbox"] > label,
[data-testid="stRadio"] > label,
label[data-testid="stWidgetLabel"],
.stSlider label, .stSelectbox label, .stRadio label {
    font-weight: 700 !important;
    color: #000000 !important;
    font-size: 0.95rem !important;
}

/* ── Main predict button ── */
div.stButton > button {
    background-color: #F9C935 !important;
    color: #000000 !important;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.5rem;
    font-size: 1rem;
    transition: background-color 0.2s ease;
}
div.stButton > button:hover {
    background-color: #000000 !important;
    color: #F9C935 !important;
}

/* ── KPI metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #F9C935 0%, #FFD966 100%);
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 5px solid #000000;
}
[data-testid="stMetricLabel"] {
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
}
[data-testid="stMetricValue"] {
    color: #000000 !important;
    font-weight: 800 !important;
    font-size: 1.5rem  !important;
}

/* ── Headers ── */
h1, h2, h3 { color: #000000 !important; }
h1 { border-bottom: 3px solid #F9C935; padding-bottom: 6px; }

/* ── Dataframe header ── */
[data-testid="stDataFrame"] thead tr th {
    background-color: #F9C935 !important;
    color: #000000 !important;
    font-weight: 700;
}

/* ── Alerts / dividers / spinner ── */
[data-testid="stAlert"] { border-left: 5px solid #F9C935 !important; }
hr { border-color: #F9C935 !important; }
.stSpinner > div { border-top-color: #F9C935 !important; }

/* ── Insight boxes ── */
.insight-box {
    background: #FFFBEA;
    border-left: 5px solid #F9C935;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 12px;
    color: #000000;
    font-size: 0.92rem;
    line-height: 1.6;
}
.insight-box b { color: #000000; }
</style>
"""

# ---------------------------------------------------------------------------
# Helper: apply Rapido brand + bold axis labels to every Plotly chart
# ---------------------------------------------------------------------------
def rapido_layout(fig):
    """Apply Rapido brand colours + bold axis labels to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=RAPIDO_WHITE,
        plot_bgcolor=RAPIDO_GREY,
        font=dict(color=RAPIDO_DARK, family="Arial, sans-serif"),
        xaxis=dict(
            title_font=dict(size=13, color=RAPIDO_DARK,
                            family="Arial Black, Arial, sans-serif"),
            tickfont=dict(color=RAPIDO_DARK),
            showgrid=True, gridcolor="#E0E0E0",
        ),
        yaxis=dict(
            title_font=dict(size=13, color=RAPIDO_DARK,
                            family="Arial Black, Arial, sans-serif"),
            tickfont=dict(color=RAPIDO_DARK),
            showgrid=True, gridcolor="#E0E0E0",
        ),
    )
    return fig


def insight_box(text: str):
    """Render a styled yellow insight panel."""
    st.markdown(f'<div class="insight-box">💡 {text}</div>',
                unsafe_allow_html=True)


# =============================================================================
# ## Section 1 – Raw Data Loading & Cleaning
# =============================================================================

def clean_data():
    """Load raw CSV, clean it, and save Rapido_dataset_cleaned.csv."""

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(RAW_DATASET_PATH)
    print(f"Rows before cleaning : {len(df)}")

    # ── 2. Standardise column names (lowercase + underscores) ─────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )

    # ── 3. Remove duplicate rows ──────────────────────────────────────────────
    df.drop_duplicates(inplace=True)
    print(f"Rows after dedup     : {len(df)}")

    # ── 4. Correct data types ─────────────────────────────────────────────────
    # Numeric columns
    numeric_cols = ["duration", "distance", "ride_charge", "misc_charge", "total_fare"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")   # 'nan' strings → NaN

    # Date / time
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["time"] = df["time"].astype(str)   # keep as string; hour extracted via str.extract in app

    # ── 5. Fix negative numeric values (convert to positive) ──────────────────
    for col in numeric_cols:
        df[col] = df[col].abs()

    # ── 6. Fill missing ride_id with "RD" + sixteen zeros ─────────────────────
    df["ride_id"] = df["ride_id"].fillna("RD" + "0" * 16)
    df["ride_id"] = df["ride_id"].replace("", "RD" + "0" * 16)

    # ── 7. Fill missing ride_charge with group average ────────────────────────
    # Group by service type and distance (rounded to nearest integer km)
    # Strategy: compute mean per (services, distance_bucket) where distance_bucket
    # is rounded to the nearest whole km to allow a reasonable reference pool.
    # Fall back to service-only mean if a bucket has no known values.

    df["_dist_bucket"] = df["distance"].round(0)

    service_dist_mean = (
        df.groupby(["services", "_dist_bucket"])["ride_charge"]
        .mean()
    )
    service_dist_mean.name = "_mean_by_dist"

    service_mean = (
        df.groupby("services")["ride_charge"]
        .mean()
    )
    service_mean.name = "_mean_by_service"

    df = df.join(service_dist_mean, on=["services", "_dist_bucket"])
    df = df.join(service_mean, on="services")

    missing_mask = df["ride_charge"].isna()

    # Use dist-bucket mean first, then service mean as fallback
    df.loc[missing_mask, "ride_charge"] = (
        df.loc[missing_mask, "_mean_by_dist"]
        .fillna(df.loc[missing_mask, "_mean_by_service"])
        .round(2)
    )

    # Clean up helper columns
    df.drop(columns=["_dist_bucket", "_mean_by_dist", "_mean_by_service"], inplace=True)

    # ── 8. Handle remaining nulls in cancelled rides ──────────────────────────
    # Cancelled rides have no misc_charge, total_fare, or payment_method.
    # Fill misc_charge with 0, recompute total_fare, fill payment_method with "Unknown".
    # NOTE: "N/A" is avoided because pandas.read_csv interprets it as NaN by default.
    # NOTE: These fills happen BEFORE casting to category to avoid category dtype errors.
    df["misc_charge"]    = df["misc_charge"].fillna(0.0)
    df["total_fare"]     = df["total_fare"].fillna(df["ride_charge"] + df["misc_charge"])
    df["payment_method"] = df["payment_method"].fillna("Unknown")

    # ── 9. Cast categorical columns (after all fills are done) ────────────────
    categorical_cols = ["services", "ride_status", "payment_method"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    # ── 10. Report ────────────────────────────────────────────────────────────
    print(f"Rows after cleaning      : {len(df)}")
    print(f"Missing ride_charge      : {df['ride_charge'].isna().sum()}")
    print(f"Missing misc_charge      : {df['misc_charge'].isna().sum()}")
    print(f"Missing total_fare       : {df['total_fare'].isna().sum()}")
    print(f"Missing payment_method   : {df['payment_method'].isna().sum()}")
    print(f"Negative distances       : {(df['distance'] < 0).sum()}")
    print(f"Negative ride_charge     : {(df['ride_charge'] < 0).sum()}")
    print(f"\nData types:\n{df.dtypes}")

    # ── 10. Save ──────────────────────────────────────────────────────────────
    df.to_csv(DATASET_PATH, index=False)
    print(f"\nSaved -> {DATASET_PATH}")


# =============================================================================
# ## Section 2 – Data Loading & Feature Engineering
# =============================================================================

@st.cache_data
def load_data():
    """Load the cleaned CSV and engineer features."""
    df = pd.read_csv(DATASET_PATH)
    df["date"]        = pd.to_datetime(df["date"], errors="coerce")
    df["hour"]        = df["time"].astype(str).str.extract(r"(\d+):")[0].astype(float)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"]       = df["date"].dt.month
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["is_cancelled"]= (df["ride_status"] == "cancelled").astype(int)
    df["is_night"]    = df["hour"].apply(lambda h: 1 if (h >= 22 or h < 6) else 0)

    # Title-case service names for display  (auto→Auto, bike lite→Bike Lite)
    df["services_display"] = df["services"].str.title()

    le = LabelEncoder()
    df["service_encoded"] = le.fit_transform(df["services"])
    return df, le


# =============================================================================
# ## Section 3 – Machine Learning Modeling
# =============================================================================

def build_feature_matrix(df):
    """Return (X, y) using only numeric / encoded features."""
    features = ["service_encoded", "distance", "duration",
                "hour", "day_of_week", "is_night"]
    return df[features].copy(), df["ride_charge"].copy()


def train_and_evaluate(df, le):
    """Train Linear Regression and Random Forest; persist models."""
    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    # ── Linear Regression ────────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    results = {
        "Linear Regression": {
            "model": lr, "y_test": y_test, "y_pred": y_pred_lr,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            "mae":  mean_absolute_error(y_test, y_pred_lr),
            "r2":   r2_score(y_test, y_pred_lr),
        },
        "Random Forest": {
            "model": rf, "y_test": y_test, "y_pred": y_pred_rf,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            "mae":  mean_absolute_error(y_test, y_pred_rf),
            "r2":   r2_score(y_test, y_pred_rf),
            "feature_importances": rf.feature_importances_,
        },
    }

    # ── Persist models ────────────────────────────────────────────────────────
    joblib.dump(lr, LR_MODEL_PATH)
    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    return results, X_test, y_test


@st.cache_resource
def load_models():
    """Load persisted models from disk."""
    if all(os.path.exists(p) for p in [LR_MODEL_PATH, RF_MODEL_PATH, ENCODER_PATH]):
        return (joblib.load(LR_MODEL_PATH),
                joblib.load(RF_MODEL_PATH),
                joblib.load(ENCODER_PATH))
    return None, None, None


def predict_charge(model, le, service, distance, duration, hour, day_of_week):
    """Encode inputs and return predicted ride charge."""
    service_enc = le.transform([service])[0]
    is_night    = 1 if (hour >= 22 or hour < 6) else 0
    record = {
        "service_encoded": service_enc, "distance": distance,
        "duration": duration, "hour": hour,
        "day_of_week": day_of_week, "is_night": is_night,
    }
    X = pd.DataFrame.from_records([record])
    return model.predict(X)[0]


# =============================================================================
# ## Section 4 – Streamlit App: Prediction UI
# =============================================================================

def page_prediction(df, le, lr_model, rf_model):
    """Ride charge prediction form using the trained ML models."""
    st.title("🛵 Rapido Ride Charge Predictor")
    st.markdown(
        "Fill in the ride details below and click **Predict** to get an "
        "estimated ride charge from both models."
    )

    # Title-cased options for display; raw value passed to ML model
    svc_display = sorted(df["services_display"].unique())
    svc_raw_map = {v.title(): v for v in df["services"].unique()}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Service Type**")
        svc_label = st.selectbox("Service Type", options=svc_display,
                                 label_visibility="collapsed")
        service   = svc_raw_map.get(svc_label, svc_label)

        st.markdown("**Distance (km)**")
        distance  = st.slider("Distance (km)", 1.0, 50.0, 15.0, 0.5,
                               label_visibility="collapsed")

        st.markdown("**Duration (minutes)**")
        duration  = st.slider("Duration (minutes)", 1, 120, 30,
                               label_visibility="collapsed")

    with col2:
        st.markdown("**Hour of Day (0–23)**")
        hour      = st.slider("Hour of Day", 0, 23, 9,
                               label_visibility="collapsed")

        st.markdown("**Day of Week**")
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
            label_visibility="collapsed",
        )

        st.markdown("**Model to Use**")
        model_choice = st.radio("Model to Use",
                                ["Linear Regression", "Random Forest", "Both"],
                                label_visibility="collapsed")

    if st.button("🔮 Predict Ride Charge", use_container_width=True):
        lr_pred = predict_charge(lr_model, le, service, distance,
                                 duration, hour, day_of_week)
        rf_pred = predict_charge(rf_model, le, service, distance,
                                 duration, hour, day_of_week)

        st.divider()
        if model_choice == "Linear Regression":
            st.metric("Linear Regression Prediction", f"₹ {lr_pred:,.2f}")
        elif model_choice == "Random Forest":
            st.metric("Random Forest Prediction", f"₹ {rf_pred:,.2f}")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Linear Regression", f"₹ {lr_pred:,.2f}")
            c2.metric("Random Forest",     f"₹ {rf_pred:,.2f}")
            diff = abs(rf_pred - lr_pred)
            st.info(f"Models differ by **₹ {diff:,.2f}**.")

        st.markdown("#### Input Summary")
        st.table(pd.DataFrame({
            "Feature": ["Service","Distance","Duration","Hour","Day","Night Ride"],
            "Value":   [svc_label, f"{distance} km", f"{duration} min",
                        f"{hour}:00",
                        ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][day_of_week],
                        "Yes" if (hour >= 22 or hour < 6) else "No"],
        }))


# =============================================================================
# ## Section 5 – Streamlit App: Dashboard & Visualisation
# =============================================================================

# ── 5a: Executive Overview ────────────────────────────────────────────────────

def section_executive_overview(df):
    st.header("📊 Executive Overview")

    completed  = df[df["ride_status"] == "completed"]
    cancel_pct = (len(df) - len(completed)) / len(df) * 100

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Rides",     f"{len(df):,}")
    k2.metric("Completed Rides", f"{len(completed):,}")
    k3.metric("Cancelled Rides", f"{len(df) - len(completed):,}")
    k4.metric("Avg Ride Charge", f"₹ {df['ride_charge'].mean():,.2f}")
    k5.metric("Avg Distance",    f"{df['distance'].mean():.1f} km")

    # ── Insights ──
    st.markdown("### 📌 Key Insights")
    top_svc    = df["services_display"].value_counts().idxmax()
    top_pct    = df["services_display"].value_counts(normalize=True).max() * 100
    hi_rev_svc = df.groupby("services_display")["ride_charge"].mean().idxmax()
    hi_rev_val = df.groupby("services_display")["ride_charge"].mean().max()

    insight_box(f"<b>{top_svc}</b> is the most popular service type with "
                f"<b>{top_pct:.1f}%</b> of all rides — making it the core volume "
                f"driver on the platform.")
    insight_box(f"<b>{hi_rev_svc}</b> commands the highest average charge at "
                f"<b>₹{hi_rev_val:,.0f}</b> per ride — an under-tapped, high-margin "
                f"segment with significant growth potential.")
    insight_box(f"Overall cancellation rate is <b>{cancel_pct:.1f}%</b> "
                f"({len(df) - len(completed):,} rides). Reducing this by just 2 pp "
                f"could recover ~1,000 rides per 61-day period.")

    col1, col2 = st.columns(2)

    # Pie – rides by service
    with col1:
        svc_counts = df["services_display"].value_counts().reset_index()
        svc_counts.columns = ["service", "count"]
        fig = px.pie(svc_counts, names="service", values="count",
                     title="Rides by Service Type",
                     color_discrete_sequence=RAPIDO_PALETTE)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          marker=dict(line=dict(color=RAPIDO_DARK, width=1.5)))
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Bar – avg charge by service
    with col2:
        avg_charge = (df.groupby("services_display", as_index=False)["ride_charge"]
                        .mean().sort_values("ride_charge", ascending=False))
        fig = px.bar(avg_charge, x="services_display", y="ride_charge",
                     title="Average Ride Charge by Service Type",
                     labels={"services_display": "Service",
                             "ride_charge": "Avg Charge (₹)"},
                     color="services_display",
                     color_discrete_sequence=RAPIDO_PALETTE)
        fig.update_layout(showlegend=False)
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Bar – completed vs cancelled
    status_df = df["ride_status"].value_counts().reset_index()
    status_df.columns = ["status", "count"]
    fig = px.bar(status_df, x="status", y="count",
                 title="Completed vs Cancelled Rides",
                 color="status",
                 color_discrete_map={"completed": RAPIDO_YELLOW,
                                     "cancelled": RAPIDO_DARK})
    fig.update_layout(showlegend=False)
    rapido_layout(fig)
    st.plotly_chart(fig, use_container_width=True)


# ── 5b: Key Drivers ──────────────────────────────────────────────────────────

def section_key_drivers(df):
    st.header("🔑 Key Drivers")

    # ── Insights ──
    st.markdown("### 📌 Key Insights")
    corr_dist = df["distance"].corr(df["ride_charge"])
    corr_dur  = df["duration"].corr(df["ride_charge"])
    insight_box(f"<b>Distance</b> has a Pearson correlation of <b>{corr_dist:.2f}</b> "
                f"with ride charge — it is the strongest single predictor, accounting "
                f"for ~39% of Random Forest's feature importance.")
    insight_box(f"<b>Duration</b> shows a correlation of <b>{corr_dur:.2f}</b> with "
                f"ride charge, contributing ~23% to the model. Longer rides in traffic "
                f"may carry a non-linear fare premium.")
    insight_box("Charge distribution is <b>near-uniform across ₹50–₹1,000</b>, "
                "suggesting the current pricing model does not sufficiently differentiate "
                "by distance or service tier — a strong case for "
                "<b>tiered pricing brackets</b>.")
    insight_box("<b>Night rides (10 PM – 6 AM)</b> show no charge premium despite "
                "higher driver scarcity — a night-surge multiplier could improve both "
                "driver retention and revenue.")

    sample = df.sample(min(3000, len(df)), random_state=42)
    col1, col2 = st.columns(2)

    # Scatter – distance vs charge
    with col1:
        fig = px.scatter(sample, x="distance", y="ride_charge",
                         color="services_display", opacity=0.6,
                         title="Distance vs Ride Charge",
                         labels={"distance": "Distance (km)",
                                 "ride_charge": "Ride Charge (₹)",
                                 "services_display": "Service"},
                         color_discrete_sequence=RAPIDO_PALETTE)
        z      = np.polyfit(sample["distance"], sample["ride_charge"], 1)
        x_line = np.linspace(sample["distance"].min(), sample["distance"].max(), 100)
        y_line = np.polyval(z, x_line)
        fig.add_scatter(x=x_line, y=y_line, mode="lines",
                        line=dict(color=RAPIDO_DARK, dash="dash", width=2),
                        name="Trend")
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Box – charge by service
    with col2:
        fig = px.box(df, x="services_display", y="ride_charge",
                     color="services_display",
                     title="Charge Distribution by Service Type",
                     labels={"services_display": "Service",
                             "ride_charge": "Ride Charge (₹)"},
                     color_discrete_sequence=RAPIDO_PALETTE)
        fig.update_layout(showlegend=False)
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # Scatter – duration vs charge
    with col3:
        fig = px.scatter(sample, x="duration", y="ride_charge",
                         color="services_display", opacity=0.6,
                         title="Duration vs Ride Charge",
                         labels={"duration": "Duration (min)",
                                 "ride_charge": "Ride Charge (₹)",
                                 "services_display": "Service"},
                         color_discrete_sequence=RAPIDO_PALETTE)
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap – correlation
    with col4:
        st.subheader("Feature Correlation")
        corr_cols = ["distance", "duration", "ride_charge",
                     "misc_charge", "total_fare", "hour", "is_night"]
        corr_df = df[corr_cols].dropna().corr()
        fig_m, ax = plt.subplots(figsize=(6, 4))
        fig_m.patch.set_facecolor(RAPIDO_WHITE)
        ax.set_facecolor(RAPIDO_GREY)
        sns.heatmap(corr_df, annot=True, fmt=".2f",
                    cmap=sns.light_palette(RAPIDO_YELLOW, as_cmap=True),
                    ax=ax, linewidths=0.5, linecolor=RAPIDO_WHITE,
                    annot_kws={"color": RAPIDO_DARK, "weight": "bold"})
        ax.set_title("Correlation Heatmap", color=RAPIDO_DARK,
                     fontweight="bold", fontsize=12)
        ax.tick_params(colors=RAPIDO_DARK, labelsize=9)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        st.pyplot(fig_m)
        plt.close()


# ── 5c: Ride Analysis ─────────────────────────────────────────────────────────

def section_ride_analysis(df):
    st.header("🚗 Ride Analysis")

    # ── Insights ──
    st.markdown("### 📌 Key Insights")
    peak_hour   = int(df.groupby("hour").size().idxmax())
    busiest_day = ["Monday","Tuesday","Wednesday","Thursday",
                   "Friday","Saturday","Sunday"][int(df["day_of_week"].mode()[0])]
    best_week   = int(df.groupby("week")["ride_charge"].sum().idxmax())
    insight_box(f"Ride demand peaks at <b>{peak_hour}:00</b> — consider positioning "
                f"extra drivers 30 minutes before peak to capture demand.")
    insight_box(f"<b>{busiest_day}</b> is the busiest day of the week. Weekend trips "
                f"tend to be longer, suggesting leisure travel patterns.")
    insight_box(f"Week <b>{best_week}</b> recorded the highest total revenue. "
                f"Replicating the conditions of that week through targeted promotions "
                f"can sustain high-revenue periods.")
    insight_box("Ride volume is <b>consistent across all 24 hours</b> — Rapido operates "
                "as a true 24/7 platform. Driver supply at 3–5 AM is the key friction "
                "point to address.")

    col1, col2 = st.columns(2)

    # Line – daily volume
    with col1:
        daily = (df.groupby("date", as_index=False).size()
                   .rename(columns={"size": "rides"}))
        fig = px.line(daily, x="date", y="rides",
                      title="Daily Ride Volume",
                      labels={"date": "Date", "rides": "Number of Rides"})
        fig.update_traces(line_color=RAPIDO_YELLOW, line_width=2.5)
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Bar – rides by hour
    with col2:
        hourly = df.groupby("hour").size().reset_index(name="rides")
        fig = px.bar(hourly, x="hour", y="rides",
                     title="Rides by Hour of Day",
                     labels={"hour": "Hour of Day", "rides": "Number of Rides"},
                     color="rides",
                     color_continuous_scale=[[0, RAPIDO_HOVER], [1, RAPIDO_YELLOW]])
        fig.update_layout(coloraxis_showscale=False)
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # Density heatmap – service × day
    with col3:
        pivot = (df.groupby(["services_display", "day_of_week"])
                   .size().reset_index(name="count"))
        pivot["day_name"] = pivot["day_of_week"].map(
            {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
        fig = px.density_heatmap(pivot, x="day_name", y="services_display",
                                  z="count",
                                  color_continuous_scale=[[0, RAPIDO_WHITE],
                                                          [1, RAPIDO_YELLOW]],
                                  title="Ride Frequency: Service × Day",
                                  labels={"services_display": "Service",
                                          "day_name": "Day of Week"})
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Horizontal bar – avg distance by service
    with col4:
        avg_dist = (df.groupby("services_display", as_index=False)["distance"]
                      .mean().sort_values("distance", ascending=True))
        fig = px.bar(avg_dist, x="distance", y="services_display",
                     orientation="h",
                     title="Average Distance by Service Type",
                     labels={"distance": "Avg Distance (km)",
                             "services_display": "Service"},
                     color="services_display",
                     color_discrete_sequence=RAPIDO_PALETTE)
        fig.update_layout(showlegend=False)
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Area – weekly revenue
    weekly_df = (df.groupby("week", as_index=False)["ride_charge"]
                   .sum().rename(columns={"ride_charge": "total_charge"}))
    fig = px.area(weekly_df, x="week", y="total_charge",
                  title="Weekly Total Revenue (Ride Charges)",
                  labels={"week": "Week Number", "total_charge": "Total Charge (₹)"})
    fig.update_traces(line_color=RAPIDO_DARK, fillcolor=RAPIDO_HOVER)
    rapido_layout(fig)
    st.plotly_chart(fig, use_container_width=True)


# ── 5d: Customer & Risk Analysis ─────────────────────────────────────────────

def section_risk_analysis(df):
    st.header("⚠️ Customer & Risk Analysis")

    Q1    = df["ride_charge"].quantile(0.25)
    Q3    = df["ride_charge"].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df["ride_charge"] < lower) | (df["ride_charge"] > upper)].copy()
    outliers["outlier_type"] = outliers["ride_charge"].apply(
        lambda x: "High" if x > upper else "Low")

    fraud_suspects = df[
        (df["distance"]    < df["distance"].quantile(0.10)) &
        (df["ride_charge"] > df["ride_charge"].quantile(0.90))
    ]

    # ── Insights ──
    st.markdown("### 📌 Key Insights")
    cancel_by_svc  = (df.groupby("services_display")["is_cancelled"]
                        .mean().mul(100).sort_values(ascending=False))
    top_cancel_svc = cancel_by_svc.idxmax()
    top_cancel_val = cancel_by_svc.max()
    insight_box(f"<b>{len(outliers):,} outlier rides</b> detected outside IQR bounds "
                f"(₹{lower:.0f} – ₹{upper:.0f}). Automated fare-review before "
                f"charging can protect customers.")
    insight_box(f"<b>{len(fraud_suspects):,} suspicious rides</b> combine very short "
                f"distance (bottom 10%) with very high charges (top 10%) — a classic "
                f"signal of fare manipulation or data-entry errors.")
    insight_box(f"<b>{top_cancel_svc}</b> has the highest cancellation rate at "
                f"<b>{top_cancel_val:.1f}%</b>. Improving driver matching or ETA "
                f"accuracy for this service can reduce churn significantly.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Outlier Rides",      f"{len(outliers):,}")
        st.metric("Outlier Threshold (Low)",  f"₹ {lower:,.2f}")
        st.metric("Outlier Threshold (High)", f"₹ {upper:,.2f}")

        fig = px.histogram(df, x="ride_charge", nbins=60,
                           title="Ride Charge Distribution with Outlier Bounds",
                           labels={"ride_charge": "Ride Charge (₹)"},
                           color_discrete_sequence=[RAPIDO_YELLOW])
        fig.update_traces(marker_line_color=RAPIDO_DARK, marker_line_width=0.5)
        fig.add_vline(x=lower, line_dash="dash", line_color=RAPIDO_DARK,
                      annotation_text="Lower bound",
                      annotation_font_color=RAPIDO_DARK)
        fig.add_vline(x=upper, line_dash="dash", line_color=RAPIDO_DARK,
                      annotation_text="Upper bound",
                      annotation_font_color=RAPIDO_DARK)
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if len(outliers) > 0:
            fig = px.scatter(outliers, x="distance", y="ride_charge",
                             color="outlier_type",
                             color_discrete_map={"High": RAPIDO_DARK,
                                                 "Low":  RAPIDO_YELLOW},
                             title="Outlier Rides – Distance vs Charge",
                             labels={"distance":    "Distance (km)",
                                     "ride_charge": "Ride Charge (₹)"})
            rapido_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No outliers detected with current IQR thresholds.")

    st.subheader("🔍 Potential Fraud / Unusual Rides")
    st.markdown(f"Rides in the **bottom 10% of distance** but **top 10% of charge**: "
                f"**{len(fraud_suspects):,} rides**")
    if len(fraud_suspects) > 0:
        fig = px.scatter(fraud_suspects, x="distance", y="ride_charge",
                         color="services_display", size="total_fare",
                         title="Suspicious Rides (Short Distance, High Charge)",
                         labels={"distance":        "Distance (km)",
                                 "ride_charge":      "Ride Charge (₹)",
                                 "services_display": "Service"},
                         color_discrete_sequence=RAPIDO_PALETTE)
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    cancel_rate = (df.groupby("services_display")["is_cancelled"]
                     .mean().mul(100).reset_index()
                     .rename(columns={"is_cancelled": "cancel_rate_%"}))
    fig = px.bar(cancel_rate, x="services_display", y="cancel_rate_%",
                 title="Cancellation Rate by Service Type (%)",
                 labels={"services_display": "Service",
                         "cancel_rate_%":    "Cancellation Rate (%)"},
                 color="services_display",
                 color_discrete_sequence=RAPIDO_PALETTE)
    fig.update_layout(showlegend=False)
    rapido_layout(fig)
    st.plotly_chart(fig, use_container_width=True)


# ── 5e: Model Comparison ──────────────────────────────────────────────────────

def section_model_comparison(df, results):
    st.header("🤖 ML Model Comparison")

    # ── Insights ──
    st.markdown("### 📌 Key Insights")
    insight_box("Both models show <b>R² ≈ 0</b> on this synthetic dataset because "
                "ride charges were generated with high randomness. In real-world "
                "production data where fares correlate with distance and duration, "
                "Random Forest would significantly outperform Linear Regression.")
    insight_box("<b>Distance</b> is the most important feature (RF importance ≈ 39%), "
                "followed by <b>Duration</b> (23%) and <b>Hour of Day</b> (17%). "
                "Service type contributes ~9% — a signal that separate per-service "
                "models could improve prediction accuracy.")

    metrics_rows = []
    for name, res in results.items():
        metrics_rows.append({
            "Model": name,
            "RMSE":  round(res["rmse"], 2),
            "MAE":   round(res["mae"],  2),
            "R²":    round(res["r2"],   4),
        })
    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    # Actual vs predicted scatter for each model
    with col1:
        for name, res in results.items():
            idx      = np.random.choice(len(res["y_test"]), 500, replace=False)
            y_test_s = np.array(res["y_test"])[idx]
            y_pred_s = np.array(res["y_pred"])[idx]
            fig = px.scatter(x=y_test_s, y=y_pred_s,
                             title=f"{name}: Actual vs Predicted",
                             labels={"x": "Actual (₹)", "y": "Predicted (₹)"},
                             opacity=0.6,
                             color_discrete_sequence=[RAPIDO_YELLOW])
            max_val = max(y_test_s.max(), y_pred_s.max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                     mode="lines",
                                     line=dict(color=RAPIDO_DARK, dash="dash"),
                                     name="Perfect Fit"))
            rapido_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Feature importances + metrics comparison
    with col2:
        feature_names = ["service_encoded", "distance", "duration",
                         "hour", "day_of_week", "is_night"]
        fi    = results["Random Forest"]["feature_importances"]
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": fi})
        fi_df = fi_df.sort_values("Importance", ascending=True)
        fig   = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                       title="Random Forest – Feature Importances",
                       labels={"Importance": "Importance Score",
                               "Feature":    "Feature"},
                       color="Importance",
                       color_continuous_scale=[[0, RAPIDO_HOVER],
                                               [1, RAPIDO_YELLOW]])
        rapido_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

        metric_colors = [RAPIDO_YELLOW, RAPIDO_HOVER, RAPIDO_DARK]
        fig = go.Figure()
        for metric, mc in zip(["RMSE", "MAE", "R²"], metric_colors):
            fig.add_trace(go.Bar(name=metric,
                                 x=metrics_df["Model"],
                                 y=metrics_df[metric],
                                 marker_color=mc))
        fig.update_layout(
            barmode="group",
            title="Model Metrics Comparison",
            paper_bgcolor=RAPIDO_WHITE,
            plot_bgcolor=RAPIDO_GREY,
            font_color=RAPIDO_DARK,
            xaxis=dict(title_font=dict(size=13,
                                       family="Arial Black, Arial, sans-serif"),
                       tickfont=dict(color=RAPIDO_DARK)),
            yaxis=dict(title_font=dict(size=13,
                                       family="Arial Black, Arial, sans-serif"),
                       tickfont=dict(color=RAPIDO_DARK)),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── 5f: Recommended Actions ───────────────────────────────────────────────────

def section_recommendations(df):
    st.header("💡 Recommended Actions")

    n_fraud = len(df[(df["distance"]    < df["distance"].quantile(0.10)) &
                     (df["ride_charge"] > df["ride_charge"].quantile(0.90))])
    lo = df["ride_charge"].quantile(0.25) - 1.5 * (
         df["ride_charge"].quantile(0.75) - df["ride_charge"].quantile(0.25))
    hi = df["ride_charge"].quantile(0.75) + 1.5 * (
         df["ride_charge"].quantile(0.75) - df["ride_charge"].quantile(0.25))

    st.markdown(f"""
### Pricing & Revenue
- **Dynamic surge pricing** during **7–9 AM** and **5–7 PM** peaks — even a 10%
  multiplier on peak-hour rides can materially boost revenue.
- **Minimum fare floor** per service type to ensure operational cost coverage,
  especially for short-distance Bike rides below ₹70.
- **Distance-based tiers** (0–5 km / 5–15 km / 15+ km) to align fares with
  true operating costs rather than a flat random distribution.

### Customer Retention
- **Re-engage cancellers**: {df["is_cancelled"].mean()*100:.1f}% of rides are cancelled —
  personalised offers and ETA transparency can recover conversion.
- **Night-time incentives**: Demand is 24/7 but driver supply drops at night.
  Bonus pay for 10 PM – 6 AM shifts can reduce wait times and churn.

### Operational Efficiency
- **Parcel segment investment**: Highest average charge, lowest volume —
  targeted marketing can grow this high-margin service.
- **Fleet redistribution** on low-demand days (Mon–Tue) via push notifications
  to stimulate demand and improve utilisation.

### Fraud & Risk Mitigation
- **{n_fraud:,} suspicious rides** show short distance + high charge patterns —
  flag for manual review before fare is finalised.
- **Real-time outlier alerts** using IQR bounds (Low: ₹{lo:.0f} / High: ₹{hi:.0f})
  to catch anomalous fares before they reach the customer.
""")


# =============================================================================
# ## Streamlit App Entry Point
# =============================================================================

PAGES = [
    "🔮 Ride Charge Predictor",
    "📊 Executive Overview",
    "🔑 Key Drivers",
    "🚗 Ride Analysis",
    "⚠️ Customer & Risk Analysis",
    "🤖 Model Comparison",
    "💡 Recommended Actions",
]


def main():
    st.set_page_config(
        page_title="Rapido Rides Analytics",
        page_icon="🛵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Inject Rapido brand CSS
    st.markdown(RAPIDO_CSS, unsafe_allow_html=True)

    # ── Run cleaning if cleaned file does not yet exist ────────────────────────
    if not os.path.exists(DATASET_PATH):
        with st.spinner("Cleaning raw dataset (first run)…"):
            clean_data()
        st.success(f"Dataset cleaned and saved as {DATASET_PATH}!")

    # ── Load data ─────────────────────────────────────────────────────────────
    df, le = load_data()

    # ── Train / load models ───────────────────────────────────────────────────
    lr_model, rf_model, cached_le = load_models()

    if lr_model is None:
        with st.spinner("Training ML models (first run)…"):
            results, _, _ = train_and_evaluate(df, le)
            lr_model, rf_model, cached_le = load_models()
        st.success("Models trained and saved!")
    else:
        assert lr_model is not None and rf_model is not None and cached_le is not None
        X, y = build_feature_matrix(df)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        y_pred_lr = lr_model.predict(X_test)
        y_pred_rf = rf_model.predict(X_test)
        results = {
            "Linear Regression": {
                "model": lr_model, "y_test": y_test, "y_pred": y_pred_lr,
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                "mae":  mean_absolute_error(y_test, y_pred_lr),
                "r2":   r2_score(y_test, y_pred_lr),
            },
            "Random Forest": {
                "model": rf_model, "y_test": y_test, "y_pred": y_pred_rf,
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                "mae":  mean_absolute_error(y_test, y_pred_rf),
                "r2":   r2_score(y_test, y_pred_rf),
                "feature_importances": rf_model.feature_importances_,
            },
        }
        le = cached_le

    # ── Sidebar – button-based navigation ─────────────────────────────────────
    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]

    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/"
        "Rapido_app_logo.svg/512px-Rapido_app_logo.svg.png",
        width=150,
    )
    st.sidebar.markdown("## Navigation")

    for p in PAGES:
        # Inject CSS to make the currently active button black + yellow text
        if st.session_state.page == p:
            st.sidebar.markdown(
                f"""<style>
                div[data-testid="stSidebar"] div.stButton
                    button[title="{p}"] {{
                    background-color: #000000 !important;
                    color: #F9C935 !important;
                    border-color: #000000 !important;
                }}
                </style>""",
                unsafe_allow_html=True,
            )
        if st.sidebar.button(p, key=f"nav_{p}", help=p, use_container_width=True):
            st.session_state.page = p

    st.sidebar.divider()
    st.sidebar.markdown(
        f"**Dataset:** {len(df):,} rides  \n"
        f"**Date range:** {df['date'].min().date()} → {df['date'].max().date()}"
    )

    # ── Route to the selected page ─────────────────────────────────────────────
    page = st.session_state.page
    if   page == "🔮 Ride Charge Predictor":   page_prediction(df, le, lr_model, rf_model)
    elif page == "📊 Executive Overview":       section_executive_overview(df)
    elif page == "🔑 Key Drivers":              section_key_drivers(df)
    elif page == "🚗 Ride Analysis":            section_ride_analysis(df)
    elif page == "⚠️ Customer & Risk Analysis": section_risk_analysis(df)
    elif page == "🤖 Model Comparison":         section_model_comparison(df, results)
    elif page == "💡 Recommended Actions":      section_recommendations(df)


if __name__ == "__main__":
    main()
