# Stock Price Prediction using LSTM

> **⚠️ Note:** This project is currently under development. Efforts are ongoing to enhance the model's performance and introduce advanced features. Stay tuned for updates!


This project provides a comprehensive framework for predicting stock prices using historical data and Long Short-Term Memory (LSTM) models. It includes functionalities for data collection, preprocessing, training, prediction, and visualization via a user-friendly Streamlit interface.

---

## Features

1. **Data Collection**
   - Downloads historical stock data for up to 500 symbols from Yahoo Finance.
   - Saves data locally in CSV format for further processing.

2. **Data Preprocessing**
   - Normalizes data using MinMaxScaler to ensure efficient training.
   - Converts time series data into sequences for LSTM input.

3. **Model Training**
   - Trains an LSTM model on the processed stock data.
   - Saves the trained model and scaler for future predictions.

4. **Prediction and Visualization**
   - Predicts future stock prices based on historical data.
   - Displays interactive charts and prediction results in a Streamlit web application.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your_username/stock-price-prediction-lstm.git
   cd stock-price-prediction-lstm
   ```

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up File Paths:**
   - Ensure the following files are in the working directory:
     - `path_to_stock_data.csv` (Downloaded stock data)
     - `scaler.pkl` (Scaler for normalization)
     - `your_model.h5` (Pre-trained LSTM model)

---

## Usage

### 1. Download Historical Data
Run the script to download historical data for selected stock symbols:
```bash
python download_data.py
```

### 2. Train the Model
Train an LSTM model on the historical data:
```bash
python train_model.py
```

### 3. Run the Streamlit App
Launch the Streamlit interface for prediction and visualization:
```bash
streamlit run app.py
```

### 4. Interact with the App
- Upload a CSV file with stock data or use downloaded data.
- Visualize historical prices.
- Predict and visualize future stock prices.

---

## Project Structure

```
├── app.py                 # Streamlit interface
├── download_data.py       # Script to download stock data
├── train_model.py         # LSTM model training script
├── path_to_stock_data.csv # Stock data file (example)
├── scaler.pkl             # MinMaxScaler object
├── your_model.h5          # Trained LSTM model
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Requirements

- Python >= 3.8
- Libraries:
  - pandas
  - numpy
  - tensorflow
  - matplotlib
  - seaborn
  - yfinance
  - scikit-learn
  - Streamlit

Install all dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Future Enhancements

- Add hyperparameter tuning for the LSTM model.
- Integrate batch processing for multiple stock symbols.
- Implement a Monte Carlo simulation for robust future predictions.
- Enhance visualization with interactive tools (e.g., Plotly).

---

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing stock data.
- The [TensorFlow](https://www.tensorflow.org/) community for the LSTM implementation guidance.
