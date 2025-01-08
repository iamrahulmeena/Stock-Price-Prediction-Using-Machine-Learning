import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# Define the paths
model_path = 'C:/Users/hp/Desktop/work/Stockplace/Models/shared_lstm_model.h5'  # Update with the correct model path
scaler_path = 'C:/Users/hp/Desktop/work/Stockplace/Models/shared_scaler.pkl'  # Update with the correct scaler path
stock_data_path = 'C:/Users/hp/Desktop/work/Stockplace/data/price_data'  # Path to the folder with price data

# Function to load the model
def load_model_from_path(model_path):
    return load_model(model_path)

# Function to load the scaler
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    if isinstance(scaler, MinMaxScaler):
        return scaler
    else:
        raise ValueError("Loaded object is not of type MinMaxScaler.")

# Function to evaluate a single stock's data
def evaluate_stock_data(stock_data, model, scaler, filename):
    print(f"Columns in {filename}: {stock_data.columns.tolist()}")
    
    # Ensure stock data is in the right format
    if all(col in stock_data.columns for col in ['Open', 'High', 'Low', 'Volume']):
        stock_data = stock_data[['Open', 'High', 'Low', 'Volume']]  # Use relevant columns
        stock_data = stock_data.values  # Convert to NumPy array

        # Check the shape of data before scaling
        print(f"Data shape before scaling: {stock_data.shape}")

        # Scale the data
        scaled_data = scaler.transform(stock_data)
        print(f"Data shape after scaling: {scaled_data.shape}")

        # Make predictions
        predictions = model.predict(scaled_data)

        return predictions
    else:
        print(f"Missing required columns in {filename}")
        return None

# Load model and scaler
model = load_model_from_path(model_path)
scaler = load_scaler(scaler_path)

# Loop through each CSV file in the directory containing price data
for filename in os.listdir(stock_data_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(stock_data_path, filename)

        try:
            # Read stock data
            stock_data = pd.read_csv(file_path)
            print(f"Evaluating: {filename}")

            # Evaluate the stock data
            predictions = evaluate_stock_data(stock_data, model, scaler, filename)
            if predictions is not None:
                print(f"Predictions for {filename}: {predictions}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
