# predict_next_day_close.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import math

# --------- USER SETTINGS ----------
TICKER = "AAPL"          # change to "TSLA" or other ticker if desired
START = "2018-01-01"
END = None               # None -> up to today
TEST_RATIO = 0.2         # fraction of data to hold out as test (time-based)
RANDOM_STATE = 42
# ---------------------------------

def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Open','High','Low','Close','Volume']]
    df.dropna(inplace=True)
    return df

def create_features(df):
    df = df.copy()
    # Basic features
    df['HL_diff'] = df['High'] - df['Low']
    df['OC_diff'] = df['Open'] - df['Close']
    # Lag features (previous day values)
    df['Close_prev1'] = df['Close'].shift(1)
    df['Volume_prev1'] = df['Volume'].shift(1)
    # Moving averages
    df['MA_3'] = df['Close'].rolling(window=3).mean()
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    # Target: next day's Close
    df['Close_next'] = df['Close'].shift(-1)
    df.dropna(inplace=True)  # drop rows created by shift/rolling
    return df

def time_train_test_split(df, test_ratio=0.2):
    n = len(df)
    split = int((1 - test_ratio) * n)
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train, test

def evaluate_and_print(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

def plot_actual_vs_pred(test_index, y_true, preds, labels, title_suffix=""):
    plt.figure(figsize=(12,6))
    plt.plot(test_index, y_true, label="Actual Close", linewidth=2)
    for pred, lab in zip(preds, labels):
        plt.plot(test_index, pred, label=lab, linewidth=1.25, alpha=0.9)
    plt.title(f"Actual vs Predicted Close {title_suffix}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # 1) Fetch data
    df = fetch_data(TICKER, START, END)
    if df.empty:
        raise SystemExit("No data downloaded; check ticker and date range.")
    print(f"Downloaded {len(df)} rows for {TICKER}")

    # 2) Feature engineering
    df_feat = create_features(df)
    features = ['Open','High','Low','Volume','HL_diff','OC_diff','Close_prev1','Volume_prev1','MA_3','MA_7']
    target = 'Close_next'

    # 3) Train/test split (time-based)
    train_df, test_df = time_train_test_split(df_feat, TEST_RATIO)
    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values

    # 4) Scale features for LinearRegression (optional but helpful)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5a) Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    pred_lr = lr.predict(X_test_scaled)
    evaluate_and_print(y_test, pred_lr, "LinearRegression")

    # 5b) Random Forest (no scaling required, but we can use same scaled features)
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)  # train on raw features
    pred_rf = rf.predict(X_test)
    evaluate_and_print(y_test, pred_rf, "RandomForest")

    # 6) Plot results (use test_df.index for x-axis)
    plot_actual_vs_pred(test_df.index, y_test, [pred_lr, pred_rf], ["LinearRegression", "RandomForest"], title_suffix=f"({TICKER})")

    # 7) Predict next day's Close using the latest available row
    last_row = df_feat.iloc[-1:]
    X_last = last_row[features].values
    X_last_scaled = scaler.transform(X_last)
    pred_next_lr = lr.predict(X_last_scaled)[0]
    pred_next_rf = rf.predict(X_last)[0]
    last_date = last_row.index[0].date()
    print(f"Last available date in dataset: {last_date}")
    print(f"Predicted next-day Close (LinearRegression): {pred_next_lr:.4f}")
    print(f"Predicted next-day Close (RandomForest):    {pred_next_rf:.4f}")

    # Optionally: show the last row features
    print("\nMost recent feature row:")
    print(last_row[features + ['Close']].T)

if __name__ == "__main__":
    main()

