import os
import joblib  # For saving and loading the scaler
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from time import sleep

# Directory to save the pre-trained models and scalers
MODEL_DIR = "C:/Users/hp/Desktop/work/stock-prediction-project_Main/models"

# Load company data from CSV
def load_company_data():
    try:
        # Load CSV with company info
        company_df = pd.read_csv(r"C:\Users\hp\Desktop\work\Stockplace\data\nifty500_symbols.csv")
        return company_df
    except FileNotFoundError:
        st.error("Company data file not found!")
        return None

# Function to fetch historical stock data from local CSV files
def fetch_local_data(stock_symbol):
    try:
        # Load historical stock data from the CSV file
        file_path = f"C:/Users/hp/Desktop/work/stock-prediction-project_Main/data/price_data/{stock_symbol}.NS_price_data.csv"
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return data
    except FileNotFoundError:
        st.error(f"Price data for {stock_symbol} not found!")
        return None

# Function to build and train the LSTM model
def build_and_train_model(stock_symbol, data, epochs=10, batch_size=32):
    # Normalize the stock prices using MinMaxScaler
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    data[['Close']] = close_scaler.fit_transform(data[['Close']])

    # Prepare the data for training
    X = []
    y = []
    for i in range(60, len(data)):
        X.append(data['Close'][i - 60:i].values)
        y.append(data['Close'][i])

    X = np.array(X)
    y = np.array(y)

    # Reshape for LSTM model
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build the LSTM model
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

    # Save the trained model and scaler
    model_save_path = os.path.join(MODEL_DIR, f"{stock_symbol}_lstm_model.h5")
    model.save(model_save_path)

    # Save the scaler
    scaler_save_path = os.path.join(MODEL_DIR, f"{stock_symbol}_scaler.pkl")
    joblib.dump(close_scaler, scaler_save_path)

    return model, close_scaler

# Function to predict stock price using a pre-trained model
def predict_stock_price(stock_symbol, data, model=None, scaler=None):
    # If model and scaler are not provided, load the pre-trained model and scaler
    if not model or not scaler:
        model_save_path = os.path.join(MODEL_DIR, f"{stock_symbol}_lstm_model.h5")
        scaler_save_path = os.path.join(MODEL_DIR, f"{stock_symbol}_scaler.pkl")

        if os.path.exists(model_save_path) and os.path.exists(scaler_save_path):
            model = load_model(model_save_path)
            scaler = joblib.load(scaler_save_path)
        else:
            st.warning(f"Model for {stock_symbol} not found. Retraining the model...")
            model, scaler = build_and_train_model(stock_symbol, data)

    # If the scaler is not fitted, fit it again to the data
    if not isinstance(scaler, MinMaxScaler):
        scaler = MinMaxScaler(feature_range=(0, 1))
        data[['Close']] = scaler.fit_transform(data[['Close']])

    data[['Close']] = scaler.transform(data[['Close']])

    # Prepare the data for prediction
    X = []
    for i in range(60, len(data)):
        X.append(data['Close'][i - 60:i].values)

    X = np.array(X)

    # Reshape for LSTM model
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Predict stock prices
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    actual_prices = scaler.inverse_transform(data['Close'][60:].values.reshape(-1, 1))

    # Calculate Mean Squared Error and prediction accuracy
    error = mean_squared_error(actual_prices, predictions)
    accuracy = 100 - np.sqrt(error)  # RMSE-based accuracy

    # Predict the next 5 days
    last_60_days = data['Close'][-60:].values.reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days)

    future_predictions = []
    for _ in range(5):
        # Reshape last 60 days data for LSTM model
        last_60_days_scaled = np.reshape(last_60_days_scaled, (1, 60, 1))
        next_day_prediction = model.predict(last_60_days_scaled)
        next_day_prediction = scaler.inverse_transform(next_day_prediction)
        
        future_predictions.append(next_day_prediction[0, 0])
        
        # Update last 60 days with the predicted price for the next day
        last_60_days_scaled = np.append(last_60_days_scaled[:, 1:, :], next_day_prediction.reshape(1, 1, 1), axis=1)

    return actual_prices, predictions, error, accuracy, model, scaler, future_predictions

# Function to plot results and enhance the table
def plot_results(actual_prices, predicted_prices, stock_symbol, data, future_predictions):
    # Get the dates for the X-axis (from the data)
    dates = data.index[-len(actual_prices):]  # Last `len(actual_prices)` dates for proper alignment
    dates = dates.date  # Remove time, only show date part

    # Create a DataFrame for Actual vs Predicted Prices
    result_df = pd.DataFrame({
        'Date': dates,
        'Actual Price': actual_prices.flatten(),
        'Predicted Price': predicted_prices.flatten(),
    })

    # Calculate Difference and Percentage Difference
    result_df['Difference'] = result_df['Actual Price'] - result_df['Predicted Price']
    result_df['% Difference'] = (abs(result_df['Difference']) / result_df['Actual Price']) * 100

    # Format Percentage Difference to show only 2 decimals and add "%" sign
    result_df['% Difference'] = result_df['% Difference'].apply(lambda x: f"{x:.2f}%")

    # Add Accuracy Column
    result_df['Accuracy %'] = 100 - result_df['% Difference'].str.rstrip('%').astype(float)

    # Reset index to remove it from the table view
    result_df = result_df.reset_index(drop=True)  # Remove index before displaying

    # Display the result in a table without index
    st.dataframe(result_df, use_container_width=True)  # Streamlit will handle full-width table

    # Display future predictions in a new table
    future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=5).date  # Future dates
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_predictions
    })
    
    # Display future prices table
    st.subheader("Predicted Stock Prices for the Next 5 Days")
    st.dataframe(future_df, use_container_width=True)

    # Plot the actual vs predicted stock prices
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual_prices, color='blue', label='Actual Prices')
    plt.plot(dates, predicted_prices, color='red', label='Predicted Prices')

    plt.title(f'{stock_symbol} Stock Price Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (in INR)', fontsize=12)
    plt.legend()
    plt.xticks(rotation=45)  # Rotate date labels for better readability
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    st.pyplot(plt)

# Streamlit UI setup
def main():
    st.title("Stock Price Prediction using LSTM")
    
    # Display note about data access only for Nifty 500 stocks
    st.write("This model has access only to Nifty 500 stock data. Please refer to the following file for the list of stocks in Nifty 500:")
    st.markdown("[Download Nifty 500 Stocks List](https://www.example.com/nifty500.csv)")

    # Load company data
    company_data = load_company_data()
    if company_data is None:
        return

    # User input for stock symbol with autocomplete suggestions
    stock_input = st.text_input("Enter Stock Symbol (e.g., TCS, AAPL, MSFT):", "")

    # Filter symbols based on user input
    if stock_input:
        filtered_symbols = company_data[company_data['Symbol'].str.contains(stock_input, case=False)]

        # If filtered symbols are available, show them
        if not filtered_symbols.empty:
            symbol_options = filtered_symbols['Symbol'].tolist()

            # Display the selectbox dynamically based on the filtered symbols
            stock_symbol = st.selectbox("Select a Stock Symbol", symbol_options)

            # Proceed with prediction if the stock symbol is selected
            if stock_symbol:
                # Right column for comparison metrics and prediction accuracy
                col1, col2 = st.columns([3, 1])

                with col2:
                    # Display loading spinner
                    with st.spinner(f"Fetching data and training model for {stock_symbol}..."):
                        sleep(2)  # Simulate loading time
                        
                    # Load local stock data
                    data = fetch_local_data(stock_symbol)

                    if data is not None:
                        actual_prices, predicted_prices, error, accuracy, model, scaler, future_predictions = predict_stock_price(stock_symbol, data)

                        st.success("Prediction Completed!")
                        st.subheader(f"Prediction Accuracy: {accuracy:.2f}%")
                        st.subheader(f"Mean Squared Error (MSE): {error:.4f}")

                        # Display the table and plot results
                        with col1:
                            plot_results(actual_prices, predicted_prices, stock_symbol, data, future_predictions)
                    else:
                        st.error("Failed to generate predictions.")
        else:
            st.warning("No symbols found. Please enter a more specific symbol.")

# Run the app
if __name__ == "__main__":
    main()
