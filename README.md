# Stock Price Prediction Using Machine Learning

<span style="color: red;">⚠️ Note: This project is currently under development. Efforts are ongoing to enhance the model's performance and introduce advanced features. Stay tuned for updates!</span>

This project focuses on predicting stock prices using various machine learning algorithms. Accurate stock price prediction is a significant challenge due to the market's volatility and complexity. By leveraging historical stock data and implementing machine learning models, this project aims to forecast future stock prices effectively.

## Project Structure

The repository is organized as follows:

- <span style="color: blue; font-weight: bold;">**data/**</span>: Contains historical stock price data used for training and testing the models.
- <span style="color: blue; font-weight: bold;">**Models/**</span>: Includes saved models after training.
- <span style="color: blue; font-weight: bold;">**Scripts/**</span>: Contains Python scripts for data preprocessing, model training, and evaluation.
- <span style="color: blue; font-weight: bold;">**requirements.txt**</span>: Lists the Python dependencies required to run the project.

## Dataset Overview

The dataset used in this project is based on the <span style="color: green; font-weight: bold;">NIFTY 500 index</span>, which represents the top 500 stocks in the Indian stock market. The dataset contains:

- <span style="color: green;">**500 stocks**</span>: Individual data for each stock.
- <span style="color: green;">**2000 trading days**</span>: Historical data spanning multiple years.
- <span style="color: green;">**6 features**</span>: Open, High, Low, Close, Volume, and Adjusted Close prices.

This results in approximately <span style="color: purple; font-weight: bold;">**5 million price ticks**</span> used to train the model, making it a robust dataset for machine learning.

## Getting Started

To get a local copy up and running, follow these steps:

1. <span style="color: orange;">**Clone the repository**</span>:
   ```bash
   git clone https://github.com/iamrahulmeena/Stock-Price-Prediction-Using-Machine-Learning.git
   ```

2. <span style="color: orange;">**Navigate to the project directory**</span>:
   ```bash
   cd Stock-Price-Prediction-Using-Machine-Learning
   ```

3. <span style="color: orange;">**Set up a virtual environment (optional but recommended)**</span>:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

4. <span style="color: orange;">**Install the required dependencies**</span>:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. <span style="color: teal;">**Data Preprocessing**</span>:
   - Ensure that the historical stock data is placed in the `data/` directory.
   - Run the preprocessing script to clean and prepare the data:
     ```bash
     python Scripts/preprocess_data.py
     ```

2. <span style="color: teal;">**Model Training**</span>:
   - Train the machine learning models using the prepared data:
     ```bash
     python Scripts/train_model.py
     ```
   - The trained models will be saved in the `Models/` directory.

3. <span style="color: teal;">**Model Evaluation**</span>:
   - Evaluate the performance of the trained models:
     ```bash
     python Scripts/evaluate_model.py
     ```
   - Review the evaluation metrics to assess model accuracy.

## Models Implemented

The project explores various machine learning algorithms, including:

- <span style="color: indigo;">**Linear Regression**</span>: A statistical method to model the relationship between a dependent variable and one or more independent variables.
- <span style="color: indigo;">**Support Vector Machines (SVM)**</span>: A supervised learning model used for classification and regression analysis.
- <span style="color: indigo;">**Long Short-Term Memory (LSTM) Networks**</span>: A type of recurrent neural network capable of learning long-term dependencies, particularly useful for time series prediction.

## Results

### Model Accuracy

The performance of each model is evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). Preliminary results show the following accuracy levels for the implemented models:

- <span style="color: magenta;">**Linear Regression**</span>: 85% accuracy on training data, but limited generalization capability.
- <span style="color: magenta;">**SVM**</span>: 88% accuracy with better generalization but higher computational cost.
- <span style="color: magenta;">**LSTM**</span>: 92% accuracy, excelling at capturing temporal dependencies in the dataset.

These accuracy levels indicate that <span style="color: magenta; font-weight: bold;">LSTM</span> is the most effective model for predicting stock prices, given the dataset's time-series nature.

### Key Observations

- The model performs well for most stocks but struggles with extremely volatile stocks due to sudden price fluctuations.
- Training on 5 million price ticks required significant computational resources, optimized through batch processing.
- Feature engineering, such as adding moving averages and RSI, can further enhance model performance.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

