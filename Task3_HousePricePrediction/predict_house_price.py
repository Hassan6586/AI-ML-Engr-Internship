#!/usr/bin/env python3
# predict_house_price.py
"""
Robust house price prediction script.
Handles CSV files with or without header row (auto-detect).
Expected columns (if header present): area, bedrooms, bathrooms, location, price
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib
import math

DATA_PATH = "data/house_prices.csv"
MODEL_OUT = "models/house_gbr.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42

EXPECTED_COLS = ['area', 'bedrooms', 'bathrooms', 'location', 'price']

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    # Try reading with header first
    df = pd.read_csv(path)
    # If the first row looks like data (no header), pandas will have used first row as header.
    # Detect: if all column names are numeric strings or don't match expected columns, treat it as headerless.
    cols_lower = [str(c).strip().lower() for c in df.columns]
    looks_like_headerless = True
    # If any expected column name appears in columns, assume header present
    for ec in EXPECTED_COLS:
        if ec in cols_lower:
            looks_like_headerless = False
            break
    if looks_like_headerless:
        # Re-read as headerless and assign expected names if shape matches
        df = pd.read_csv(path, header=None)
        if df.shape[1] == len(EXPECTED_COLS):
            df.columns = EXPECTED_COLS
            print("Detected headerless CSV. Assigned columns:", EXPECTED_COLS)
        else:
            print("WARNING: File seems headerless but has", df.shape[1], "columns.")
            print("Current columns:", list(df.columns))
            print("Expected columns (recommended):", EXPECTED_COLS)
            # try to proceed by naming columns generically
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
            print("Assigned fallback column names:", list(df.columns))
    else:
        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]
    return df

def validate_and_prepare(df):
    # Ensure we have a price column (target)
    lower_cols = [c.lower() for c in df.columns]
    price_col = None
    for candidate in ['price', 'Price', 'saleprice', 'SalePrice']:
        if candidate.lower() in lower_cols:
            price_col = df.columns[lower_cols.index(candidate.lower())]
            break
    if price_col is None:
        raise ValueError(f"Could not find target column (price). Found columns: {list(df.columns)}. "
                         f"Please include a 'price' column or rename your column to 'price'.")
    # If numeric columns are named oddly, attempt to detect common numeric columns
    # try to find area, bedrooms, bathrooms
    def find_col_like(possible_names):
        for name in possible_names:
            if name in lower_cols:
                return df.columns[lower_cols.index(name)]
        return None

    area_col = find_col_like(['area', 'square_feet', 'sqft', 'size'])
    bed_col = find_col_like(['bedrooms', 'beds', 'bedroom'])
    bath_col = find_col_like(['bathrooms', 'baths', 'bath'])

    # If we assigned headerless expected names earlier, these will exist
    if area_col is None and 'area' in df.columns:
        area_col = 'area'
    if bed_col is None and 'bedrooms' in df.columns:
        bed_col = 'bedrooms'
    if bath_col is None and 'bathrooms' in df.columns:
        bath_col = 'bathrooms'

    required = {'area': area_col, 'bedrooms': bed_col, 'bathrooms': bath_col}
    missing = [k for k,v in required.items() if v is None]
    if missing:
        print("WARNING: Could not automatically find these columns:", missing)
        print("Detected columns:", list(df.columns))
        # proceed but raise if too few numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 1:
            raise ValueError("Not enough numeric features detected. Please ensure your CSV has numeric feature columns.")
    # Return names
    return df, price_col, area_col, bed_col, bath_col

def preprocess(df, area_col, bed_col, bath_col, price_col):
    # Keep only useful columns if present
    cols_to_keep = []
    for c in [area_col, bed_col, bath_col, 'location', price_col]:
        if c and c in df.columns:
            cols_to_keep.append(c)
    df = df[cols_to_keep].copy()
    # Fill missing numeric with median
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c].fillna(df[c].median(), inplace=True)
    # Categorical fill
    if 'location' in df.columns:
        df['location'] = df['location'].astype(str).fillna('Unknown')
        df = pd.get_dummies(df, columns=['location'], drop_first=True)
    # Ensure target is numeric
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
    return df

def train_and_eval(df, price_col):
    X = df.drop(columns=[price_col])
    y = df[price_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    pred_lr = lr.predict(X_test_scaled)

    gbr = GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE)
    gbr.fit(X_train, y_train)  # GBR works on raw X (but scaling isn't harmful)
    pred_gbr = gbr.predict(X_test)

    mae_lr = mean_absolute_error(y_test, pred_lr)
    rmse_lr = math.sqrt(mean_squared_error(y_test, pred_lr))
    mae_gbr = mean_absolute_error(y_test, pred_gbr)
    rmse_gbr = math.sqrt(mean_squared_error(y_test, pred_gbr))

    print(f"LinearRegression -> MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}")
    print(f"GradientBoosting -> MAE: {mae_gbr:.2f}, RMSE: {rmse_gbr:.2f}")

    # Plot actual vs predicted for GBR
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=pred_gbr, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted (GradientBoosting)")
    plt.tight_layout()
    plt.savefig("images/actual_vs_predicted.png")
    plt.close()

    # Feature importance (if model has)
    try:
        feature_names = X.columns.tolist()
        importances = gbr.feature_importances_
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(8,6))
        fi.head(15).sort_values().plot(kind='barh')
        plt.title("Feature Importances (Top 15)")
        plt.tight_layout()
        plt.savefig("images/feature_importances.png")
        plt.close()
        print("\nTop feature importances:\n", fi.head(15))
    except Exception as e:
        print("Could not compute feature importances:", e)

    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(gbr, MODEL_OUT)
    joblib.dump(scaler, "models/scaler.joblib")
    print(f"\nSaved model -> {MODEL_OUT}")

def main():
    df = load_data(DATA_PATH)
    df, price_col, area_col, bed_col, bath_col = validate_and_prepare(df)
    df_p = preprocess(df, area_col, bed_col, bath_col, price_col)
    train_and_eval(df_p, price_col)

if __name__ == "__main__":
    main()

