"""
pipeline.py

Core ML pipeline functions — load, clean, engineer, split, preprocess, train, evaluate.
Used by app.py (Streamlit UI). Kept separate so the logic is testable and reusable
outside of Streamlit if needed.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

CHICAGO_CENTER = (41.8781, -87.6298)
CATEGORICAL_COLS = ["neighbourhood", "room_type", "min_nights_bin"]


# ── 1. load ──────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw Airbnb listings CSV from disk.
    Returns a DataFrame with all original columns intact.
    """
    return pd.read_csv(filepath)


# ── 2. clean ─────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unusable columns, handle nulls, remove outliers, and log-transform price.

    Decisions:
    - Drop id/name/host_id/host_name/license/neighbourhood_group — not predictive
    - Drop rows with no price — can't train without a target
    - Cap at 99th percentile — extreme outliers ($50k listings) skew the model
    - Log-transform price — raw price is right-skewed
    """
    df = df.drop(columns=["id", "name", "host_id", "host_name", "license", "neighbourhood_group"])
    df = df.dropna(subset=["price"]).reset_index(drop=True)

    df["has_been_reviewed"] = df["last_review"].notna().astype(int)
    df["days_since_last_review"] = (
        pd.to_datetime("today") - pd.to_datetime(df["last_review"], errors="coerce")
    ).dt.days.fillna(0).astype(int)
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    upper = df["price"].quantile(0.99)
    df = df[df["price"] <= upper]

    df["price_log"] = np.log1p(df["price"])
    df = df.drop(columns=["last_review"]).reset_index(drop=True)

    return df


# ── 3. engineer features ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features and drop columns they replace.

    Decisions:
    - dist_to_center — proximity to downtown captured in one number
    - min_nights_bin — stay-type category matters more than the raw number
    """
    df = df.copy()

    df["dist_to_center"] = np.sqrt(
        (df["latitude"] - CHICAGO_CENTER[0]) ** 2 +
        (df["longitude"] - CHICAGO_CENTER[1]) ** 2
    )

    df["min_nights_bin"] = pd.cut(
        df["minimum_nights"],
        bins=[0, 1, 3, 7, 30, 365],
        labels=["nightly", "short", "weekly", "monthly", "longterm"]
    )

    df = df.drop(columns=["latitude", "longitude", "minimum_nights"]).reset_index(drop=True)
    return df


# ── 4. split ─────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split on price_log bins.
    Ensures price distribution is representative in both sets.
    Returns (train_set, test_set) as DataFrames with all columns.
    """
    df = df.reset_index(drop=True)
    df["price_log_cat"] = pd.cut(df["price_log"], bins=5, labels=False)

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_idx, test_idx in split.split(df, df["price_log_cat"]):
        train_set = df.loc[train_idx].drop(columns=["price_log_cat"])
        test_set = df.loc[test_idx].drop(columns=["price_log_cat"])

    return train_set, test_set


# ── 5. preprocess ────────────────────────────────────────────────────────────

def preprocess(train_set: pd.DataFrame, test_set: pd.DataFrame):
    """
    Separate X/y, one-hot encode categoricals, split train into train/val.
    Fit encoding on train only to prevent data leakage.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    y_train_full = train_set["price_log"]
    X_train_full = train_set.drop(columns=train_set.filter(regex="^price").columns)

    y_test = test_set["price_log"]
    X_test = test_set.drop(columns=test_set.filter(regex="^price").columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, drop_first=True)
    X_val = pd.get_dummies(X_val, columns=CATEGORICAL_COLS, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS, drop_first=True)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 6. train ─────────────────────────────────────────────────────────────────

def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train Linear Regression and Random Forest on the training set.
    Returns (model_lr, model_rf).
    """
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    return model_lr, model_rf


# ── 7. evaluate ──────────────────────────────────────────────────────────────

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """
    Return R² on train/val/test and RMSE in both log and dollar scale as a dict.
    """
    y_pred_log = model.predict(X_test)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    rmse_dollars = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_log)))

    return {
        "train_r2": model.score(X_train, y_train),
        "val_r2": model.score(X_val, y_val),
        "test_r2": model.score(X_test, y_test),
        "rmse_log": rmse_log,
        "rmse_dollars": rmse_dollars,
    }


# ── 8. predict ───────────────────────────────────────────────────────────────

def predict_price(model_lr, model_rf, listing_input: dict, train_columns: list) -> dict:
    """
    Predict nightly price in dollars from a listing's raw features.
    Applies the same feature engineering as engineer_features() to a single row.
    """
    listing = listing_input.copy()

    listing["dist_to_center"] = np.sqrt(
        (listing.pop("latitude") - CHICAGO_CENTER[0]) ** 2 +
        (listing.pop("longitude") - CHICAGO_CENTER[1]) ** 2
    )

    min_nights = listing.pop("minimum_nights")
    if min_nights <= 1:
        listing["min_nights_bin"] = "nightly"
    elif min_nights <= 3:
        listing["min_nights_bin"] = "short"
    elif min_nights <= 7:
        listing["min_nights_bin"] = "weekly"
    elif min_nights <= 30:
        listing["min_nights_bin"] = "monthly"
    else:
        listing["min_nights_bin"] = "longterm"

    input_df = pd.DataFrame([listing])
    input_df = pd.get_dummies(input_df, columns=CATEGORICAL_COLS, drop_first=True)
    input_df = input_df.reindex(columns=train_columns, fill_value=0)

    lr_pred = float(np.expm1(model_lr.predict(input_df)[0]))
    rf_pred = float(np.expm1(model_rf.predict(input_df)[0]))

    return {
        "linear_regression": round(lr_pred, 2),
        "random_forest": round(rf_pred, 2),
    }