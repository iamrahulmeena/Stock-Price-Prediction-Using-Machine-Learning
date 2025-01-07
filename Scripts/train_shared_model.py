import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

# Directory to save the pre-trained models and scalers
MODEL_DIR = "C:/Users/hp/Desktop/work/Stockplace/Models"  # Correct path for saving the model
DATA_DIR = "C:/Users/hp/Desktop/work/Stockplace/data/price_data"

# Load company data from CSV
# Function to load company data from the CSV file and extract the stock symbols
def load_company_data():
    try:
        # Check if the current working directory is correct
        print("Current working directory:", os.getcwd())
        
        # Load CSV with company info (list of Nifty 500 symbols)
        company_df = pd.read_csv("C:/Users/hp/Desktop/work/Stockplace/data/nifty500_symbols.csv")
        
        # Extract the 'Symbol' column which contains the stock symbols
        symbols = company_df['Symbol'].tolist()  # Convert to a list of symbols
        return symbols
    except FileNotFoundError:
        print("Company data file not found!")
        return None
    except KeyError:
        print("The expected 'Symbol' column is missing in the file!")
        return None

# Function to fetch historical stock data from local CSV files
def fetch_local_data(stock_symbol):
    try:
        # Load historical stock data from the CSV file
        file_path = f"{DATA_DIR}/{stock_symbol}.NS_price_data.csv"
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return data
    except FileNotFoundError:
        print(f"Price data for {stock_symbol} not found!")
        return None

# Function to fetch combined data for all stocks
def fetch_combined_data(symbols):
    all_data = []
    
    for stock_symbol in symbols:
        data = fetch_local_data(stock_symbol)
        if data is not None:
            data['Symbol'] = stock_symbol  # Add a column for the stock symbol
            all_data.append(data)
    
    # Concatenate all the data into a single DataFrame
    combined_data = pd.concat(all_data)
    return combined_data

# Function to preprocess the combined data
def preprocess_combined_data(combined_data):
    # Normalize the closing prices for all stocks using a single scaler
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    combined_data[['Close']] = close_scaler.fit_transform(combined_data[['Close']])

    # Prepare the data for training (using the same method as before)
    X = []
    y = []

    for i in range(60, len(combined_data)):
        X.append(combined_data['Close'][i - 60:i].values)
        y.append(combined_data['Close'][i])

    X = np.array(X)
    y = np.array(y)

    # Reshape for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, close_scaler

# Function to build and train the LSTM model for all stocks
def build_and_train_shared_model(X, y, epochs=10, batch_size=32):
    # Build the LSTM model for all stocks
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])

    return model

def main():
    # Load company data (Nifty 500 symbols)
    company_data = load_company_data()
    if company_data is None:
        return

    # Fetch combined data for all stocks
    combined_data = fetch_combined_data(company_data)
    if combined_data is None:
        return

    # Preprocess the combined data
    X, y, close_scaler = preprocess_combined_data(combined_data)

    # Build and train the shared model
    print("Training the shared model...")
    model = build_and_train_shared_model(X, y)
    print("Shared model trained successfully!")

    # Save the model
    model_save_path = os.path.join(MODEL_DIR, "shared_lstm_model.h5")
    model.save(model_save_path)

    # Save the scaler
    scaler_save_path = os.path.join(MODEL_DIR, "shared_scaler.pkl")
    joblib.dump(close_scaler, scaler_save_path)

if __name__ == "__main__":
    main()