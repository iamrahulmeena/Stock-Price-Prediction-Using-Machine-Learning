import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Path to your stock data directory (this should contain all 500 CSV files)
data_directory = r"C:\Users\hp\Desktop\work\Stockplace\data\price_data"  # Replace with the actual directory

# List all CSV files in the directory
csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]

# Initialize empty dataframe to hold all the data
all_data = pd.DataFrame()

# Loop through all CSV files and load the data
for file in csv_files:
    file_path = os.path.join(data_directory, file)
    stock_data = pd.read_csv(file_path)
    all_data = pd.concat([all_data, stock_data], axis=0, ignore_index=True)

# Extract features (adjust as per your dataset)
features = all_data[['Open', 'High', 'Low', 'Volume']]  # Replace with your actual feature columns

# Initialize the scaler and scale the data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Save the scaler
scaler_path = r"C:\Users\hp\Desktop\work\Stockplace\Models\shared_scaler.pkl"
joblib.dump(scaler, scaler_path)

print(f"Scaler saved at {scaler_path}")
