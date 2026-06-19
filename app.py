"""
app.py

Streamlit app walking through the Airbnb price prediction pipeline step by step.
Designed for a teaching video — each tab corresponds to one stage of the pipeline.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import (
    load_data,
    clean_data,
    engineer_features,
    split_data,
    preprocess,
    train_models,
    evaluate_model,
    predict_price,
)

st.set_page_config(page_title="Airbnb Price Predictor", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "chicago-listings-sept-2025.csv")


# ── cache expensive steps so the app stays fast as you click between tabs ───

@st.cache_data
def get_raw_data():
    return load_data(DATA_PATH)


@st.cache_data
def get_clean_data(_df_raw):
    return clean_data(_df_raw)


@st.cache_data
def get_engineered_data(_df_clean):
    return engineer_features(_df_clean)


@st.cache_resource
def get_trained_models(_df_engineered):
    train_set, test_set = split_data(_df_engineered)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(train_set, test_set)
    model_lr, model_rf = train_models(X_train, y_train)
    return model_lr, model_rf, X_train, X_val, X_test, y_train, y_val, y_test


# ── load + run pipeline once, shared across all tabs ─────────────────────────

df_raw = get_raw_data()
df_clean = get_clean_data(df_raw)
df_engineered = get_engineered_data(df_clean)
model_lr, model_rf, X_train, X_val, X_test, y_train, y_val, y_test = get_trained_models(df_engineered)

st.title("🏠 Chicago Airbnb Price Predictor")
st.caption(
    "A property management company wants to know: can we predict a fair nightly "
    "price from listing details alone? This app walks through the full ML pipeline."
)

tabs = st.tabs([
    "1. Raw Data",
    "2. Cleaning",
    "3. Explore",
    "4. Features",
    "5. Train & Evaluate",
    "6. Predict",
])


# ── 1. raw data ────────────────────────────────────────────────────────────

with tabs[0]:
    st.header("Raw Data")
    st.markdown(
        "This is the unmodified dataset — Inside Airbnb listings for Chicago, "
        "September 2025."
    )
    st.write(f"**{len(df_raw):,} rows, {df_raw.shape[1]} columns**")
    st.dataframe(df_raw.head(20))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Column types")
        st.dataframe(df_raw.dtypes.astype(str).rename("dtype"))
    with col2:
        st.subheader("Missing values")
        st.dataframe(df_raw.isnull().sum().rename("nulls"))


# ── 2. cleaning ────────────────────────────────────────────────────────────

with tabs[1]:
    st.header("Cleaning")
    st.markdown(
        """
        Steps applied in `clean_data()`:
        - Drop columns that can't help prediction: `id`, `name`, `host_id`, `host_name`, `license`, `neighbourhood_group`
        - Drop rows with no `price` — can't train without a target
        - Create `has_been_reviewed` and `days_since_last_review` from `last_review`
        - Fill missing `reviews_per_month` with 0
        - Cap price at the 99th percentile to remove outliers
        - Log-transform price → `price_log`
        """
    )

    upper = df_raw["price"].quantile(0.99)
    outliers = df_raw[df_raw["price"] > upper]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows before cleaning", f"{len(df_raw):,}")
    col2.metric("Rows after cleaning", f"{len(df_clean):,}")
    col3.metric("99th percentile price cutoff", f"${upper:,.0f}")

    st.subheader("Outlier listings removed")
    st.dataframe(
        outliers[["neighbourhood", "room_type", "price"]]
        .sort_values("price", ascending=False)
        .head(10)
    )

    st.subheader("Cleaned data preview")
    st.dataframe(df_clean.head(10))


# ── 3. explore ─────────────────────────────────────────────────────────────

with tabs[2]:
    st.header("Explore")
    st.markdown(
        "Before engineering features, let's understand what the data is telling us. "
        "This is what motivates the feature decisions in the next tab."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price distribution (log scale)")
        fig, ax = plt.subplots(figsize=(6, 4))
        df_clean["price_log"].hist(bins=40, ax=ax, color="#10b981")
        ax.set_xlabel("log(price)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Listings by neighbourhood")
        fig, ax = plt.subplots(figsize=(6, 4))
        df_clean["neighbourhood"].value_counts().head(15).plot(kind="barh", ax=ax, color="#10b981")
        ax.invert_yaxis()
        st.pyplot(fig)

    st.subheader("Geographic price distribution")
    st.caption("Color = log price, size = number of reviews")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df_clean["longitude"], df_clean["latitude"],
        c=df_clean["price_log"], cmap="jet",
        s=df_clean["number_of_reviews"] / 10 + 5,
        alpha=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="log(price)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    st.pyplot(fig)

    st.subheader("Feature correlation (pre-engineering)")
    corr = df_clean.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, cmap="RdYlGn", center=0, square=True, linewidths=2, ax=ax)
    st.pyplot(fig)


# ── 4. features ────────────────────────────────────────────────────────────

with tabs[3]:
    st.header("Feature Engineering")
    st.markdown(
        """
        Based on what we saw in the explore tab:
        - **`dist_to_center`** — the map showed price clusters around downtown;
          a single distance value captures this better than raw lat/lon
        - **`min_nights_bin`** — whether a listing targets nightly, weekly, or
          monthly stays matters more than the raw number
        - Raw `latitude`, `longitude`, `minimum_nights` are dropped, replaced by the above
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("dist_to_center distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        df_engineered["dist_to_center"].hist(bins=40, ax=ax, color="#10b981")
        ax.set_xlabel("distance to city center")
        st.pyplot(fig)

    with col2:
        st.subheader("min_nights_bin counts")
        fig, ax = plt.subplots(figsize=(6, 4))
        df_engineered["min_nights_bin"].value_counts().plot(kind="bar", ax=ax, color="#10b981")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    st.subheader("Engineered data preview")
    st.dataframe(df_engineered.head(10))


# ── 5. train & evaluate ───────────────────────────────────────────────────

with tabs[4]:
    st.header("Train & Evaluate")
    st.markdown(
        "Two models trained on the same data: Linear Regression as an "
        "interpretable baseline, Random Forest as a more powerful comparison."
    )

    eval_lr = evaluate_model(model_lr, X_train, y_train, X_val, y_val, X_test, y_test)
    eval_rf = evaluate_model(model_rf, X_train, y_train, X_val, y_val, X_test, y_test)

    comparison_df = pd.DataFrame([
        {"Model": "Linear Regression", **eval_lr},
        {"Model": "Random Forest", **eval_rf},
    ]).set_index("Model")

    st.subheader("Model comparison")
    st.dataframe(
        comparison_df.style.format({
            "train_r2": "{:.3f}", "val_r2": "{:.3f}", "test_r2": "{:.3f}",
            "rmse_log": "{:.3f}", "rmse_dollars": "${:.2f}",
        })
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Random Forest Test R²", f"{eval_rf['test_r2']:.3f}")
        st.caption(
            f"Train R² is {eval_rf['train_r2']:.3f} — the gap to test R² shows "
            "some overfitting, typical for Random Forest."
        )
    with col2:
        st.metric("Linear Regression Test R²", f"{eval_lr['test_r2']:.3f}")
        st.caption(
            "Lower than Random Forest — the relationship between features and "
            "price isn't fully linear."
        )

    st.subheader("Feature importance (Random Forest)")
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model_rf.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#10b981")
    ax.invert_yaxis()
    ax.set_xlabel("importance")
    st.pyplot(fig)

    st.info(
        "**What's missing:** guest ratings, amenities, and bedroom/bathroom counts "
        "aren't in this dataset but likely explain much of the remaining variance."
    )


# ── 6. predict ─────────────────────────────────────────────────────────────

with tabs[5]:
    st.header("Predict a Price")
    st.markdown("Enter listing details to get a predicted nightly price from both models.")

    neighbourhoods = sorted(df_clean["neighbourhood"].unique().tolist())
    room_types = sorted(df_clean["room_type"].unique().tolist())

    col1, col2, col3 = st.columns(3)

    with col1:
        neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)
        room_type = st.selectbox("Room type", room_types)
        minimum_nights = st.number_input("Minimum nights", min_value=1, value=2)

    with col2:
        latitude = st.number_input("Latitude", value=41.8827, format="%.4f")
        longitude = st.number_input("Longitude", value=-87.6233, format="%.4f")
        availability_365 = st.slider("Availability (days/year)", 0, 365, 200)

    with col3:
        number_of_reviews = st.number_input("Total reviews", min_value=0, value=10)
        reviews_per_month = st.number_input("Reviews per month", min_value=0.0, value=1.5, step=0.1)
        number_of_reviews_ltm = st.number_input("Reviews last 12 months", min_value=0, value=5)

    calculated_host_listings_count = st.number_input("Host's total listings", min_value=1, value=1)
    has_been_reviewed = st.radio("Has this listing been reviewed?", ["Yes", "No"], horizontal=True)
    days_since_last_review = st.number_input("Days since last review", min_value=0, value=30)

    if st.button("Predict price", type="primary"):
        listing_input = {
            "neighbourhood": neighbourhood,
            "latitude": latitude,
            "longitude": longitude,
            "room_type": room_type,
            "minimum_nights": minimum_nights,
            "number_of_reviews": number_of_reviews,
            "reviews_per_month": reviews_per_month,
            "calculated_host_listings_count": calculated_host_listings_count,
            "availability_365": availability_365,
            "number_of_reviews_ltm": number_of_reviews_ltm,
            "has_been_reviewed": 1 if has_been_reviewed == "Yes" else 0,
            "days_since_last_review": days_since_last_review,
        }

        result = predict_price(model_lr, model_rf, listing_input, X_train.columns.tolist())

        col1, col2 = st.columns(2)
        col1.metric("Random Forest", f"${result['random_forest']:.2f}")
        col2.metric("Linear Regression", f"${result['linear_regression']:.2f}")
        st.caption("Random Forest is the stronger model (higher test R²) — treat it as the primary estimate.")