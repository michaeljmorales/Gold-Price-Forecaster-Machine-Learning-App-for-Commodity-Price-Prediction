import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# 1. Dataset Acquisition
def get_gold_data(start_date, end_date):
    print(f"Downloading Gold price data from {start_date} to {end_date}...")
    # GC=F is the ticker for Gold Futures on Yahoo Finance
    ticker = 'GC=F'
    data = yf.download(ticker, start=start_date, end=end_date)
    # Flatten MultiIndex columns returned by newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

# 2. Preprocessing & Feature Engineering
def preprocess_data(data):
    print("Preprocessing data...")
    # Handle missing values
    data = data.dropna()
    
    # We will predict the 'Close' price of the next day
    # Create features based on historical prices
    
    # 5-day and 20-day Simple Moving Average (SMA)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Daily Return
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Volatility (Rolling standard deviation)
    data['Volatility_5'] = data['Daily_Return'].rolling(window=5).std()
    
    # Target the 'Daily_Return' shifted by -1 (next day's return) instead of raw absolute prices.
    # This completely fixes the Random Forest issue where it can't predict values higher than what it saw in training.
    data['Target_Next_Return'] = data['Close'].pct_change().shift(-1)
    
    # Drop NaNs introduced by rolling and shifting
    data = data.dropna()
    
    return data

# 3. Model Training and Evaluation Pipeline
def run_pipeline():
    # Define date range
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d') # Last 5 years
    
    # Get and preprocess data
    df = get_gold_data(start_date, end_date)
    
    # Check if download was successful
    if df.empty:
        print("Failed to download data. Exiting.")
        return
        
    df_processed = preprocess_data(df.copy())
    
    # Define features (X) and target (y)
    # Using 'Open', 'High', 'Low', 'Volume', 'Close', SMA_5, SMA_20, Daily_Return, Volatility_5
    features = ['Open', 'High', 'Low', 'Volume', 'Close', 'SMA_5', 'SMA_20', 'Daily_Return', 'Volatility_5']
    
    # Standard Index
    X = df_processed[features]
    y = df_processed['Target_Next_Return']
    
    # Split data chronologically (do not shuffle time series data)
    print("Splitting data into train and test sets...")
    test_size = 0.2
    split_index = int(len(df_processed) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Model Training
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Prediction
    print("Making predictions on test set...")
    y_pred_return = model.predict(X_test)
    
    # Reconstruct actual prices from the predicted returns
    y_pred = X_test['Close'].values * (1 + y_pred_return)
    y_test_price = X_test['Close'].values * (1 + y_test)
    
    # Evaluation
    mse = mean_squared_error(y_test_price, y_pred)
    mae = mean_absolute_error(y_test_price, y_pred)
    r2 = r2_score(y_test_price, y_pred)
    
    # Swap y_test to true reconstructed prices so the visualizer plots correctly
    y_test = pd.Series(y_test_price, index=y_test.index)
    
    print("\n--- Model Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    # Visualization
    print("\nGenerating visualization...")
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    
    test_dates = df_processed.index[split_index:]
    
    plt.plot(test_dates, y_test.values, label='Actual Gold Price', color='blue')
    plt.plot(test_dates, y_pred, label='Predicted Gold Price (Next Day)', color='orange', alpha=0.8)
    
    plt.title('Gold Price Prediction using Random Forest', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Gold Price (USD)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'gold_price_prediction.png'
    plt.savefig(plot_filename)
    print(f"Plot saved successfully as '{plot_filename}'")
    
    # Predict tomorrow's price (using today's data)
    latest_data = X.iloc[-1:].copy()
    tomorrow_prediction_return = model.predict(latest_data)
    tomorrow_prediction = latest_data['Close'].values[0] * (1 + tomorrow_prediction_return[0])
    
    print(f"\n--- Future Prediction ---")
    print(f"Current Date Extrapolated Price base: ${latest_data['Close'].values[0]:.2f}")
    print(f"Predicted target Close price for next trading day: ${tomorrow_prediction:.2f}")

if __name__ == "__main__":
    run_pipeline()
