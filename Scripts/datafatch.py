import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import os

# Load the Nifty 500 symbols
file_path = r"C:\Users\hp\Desktop\work\Stockplace\data\nifty500_symbols.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at: {file_path}")

nifty500_symbols_df = pd.read_csv(file_path)
print(nifty500_symbols_df.head())  # To confirm successful loading

# Extract the stock symbols (NSE format requires .NS suffix)
nifty500_symbols = nifty500_symbols_df['Symbol'].apply(lambda x: x + '.NS').tolist()

# Function to get historical price data for a list of stocks
def get_historical_data(symbols, start_date='2015-01-01', end_date=None):
    if end_date is None:
        end_date = dt.date.today().strftime('%Y-%m-%d')  # Fixed reference to dt

    stock_data = {}
    for symbol in symbols:
        try:
            print(f"Downloading data for {symbol}")
            stock_data[symbol] = yf.download(symbol, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
    return stock_data

# Get price data for Nifty 500 stocks
nifty500_price_data = get_historical_data(nifty500_symbols[:500])

# Save historical price data
output_dir = r"C:\Users\hp\Desktop\work\Stockplace\data\price_data"
os.makedirs(output_dir, exist_ok=True)

for symbol, df in nifty500_price_data.items():
    file_name = os.path.join(output_dir, f"{symbol}_price_data.csv")
    df.to_csv(file_name)
