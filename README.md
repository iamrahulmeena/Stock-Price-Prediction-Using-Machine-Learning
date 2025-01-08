
# Stock Price Prediction Using Machine Learning

⚠️ **Note**: *This project is currently under development. Efforts are ongoing to enhance the model's performance and introduce advanced features. Stay tuned for updates!*

---

## 📌 Project Overview

This project focuses on predicting stock prices using various machine learning algorithms. Accurate stock price prediction is a significant challenge due to the market's volatility and complexity. By leveraging historical stock data and implementing machine learning models, this project aims to forecast future stock prices effectively.

---

## 📂 Project Structure

The repository is organized as follows:

- **`data/`**: 📈 Contains historical stock price data used for training and testing the models.
- **`Models/`**: 💾 Includes saved models after training.
- **`Scripts/`**: 🛠️ Contains Python scripts for data preprocessing, model training, and evaluation.
- **`requirements.txt`**: 📋 Lists the Python dependencies required to run the project.

---

## 📊 Dataset Overview

This project uses data from the **NIFTY 500 index**, which represents the top 500 stocks in the Indian stock market.

- 📌 **500 stocks**: Individual data for each stock.
- 📌 **2000+ trading days**: Historical data spanning multiple years.
- 📌 **6 features**: `Open`, `High`, `Low`, `Close`, `Volume`, and `Adjusted Close` prices.

This results in approximately **5 million+ price ticks** being used to train the model, ensuring a robust dataset for machine learning.

---

## 🚀 Getting Started

To set up and run this project locally, follow these steps:

### Step 1: Clone the repository
```bash
git clone https://github.com/iamrahulmeena/Stock-Price-Prediction-Using-Machine-Learning.git
cd Stock-Price-Prediction-Using-Machine-Learning
```

### Step 2: Set up a virtual environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage

### 1️⃣ Data Preprocessing
Prepare and clean the historical stock data:
```bash
python Scripts/preprocess_data.py
```

### 2️⃣ Train the Model
Train machine learning models using the preprocessed data:
```bash
python Scripts/train_model.py
```

### 3️⃣ Evaluate the Model
Evaluate the performance of the trained models:
```bash
python Scripts/evaluate_model.py
```

---

## 🧠 Models Implemented

This project uses the following machine learning algorithms:

- **📘 Linear Regression**: A statistical approach to modeling the relationship between variables.
- **📗 Support Vector Machines (SVM)**: Effective for classification and regression tasks.
- **📙 Long Short-Term Memory (LSTM)**: A type of RNN suitable for time-series data.

---

## 📈 Results

### Model Accuracy
| Model                  | Accuracy (%) | Notes                        |
|------------------------|--------------|------------------------------|
| **Linear Regression**  | 63%          | Limited generalization       |
| **SVM**                | 79%          | Better generalization        |
| **LSTM**               | 93%          | Excels in capturing patterns |

💡 **Key Observations**:
- The **LSTM model** performs best for stock price prediction due to its ability to process time-series data.
- Feature engineering, like adding moving averages or RSI, could improve performance further.

---

## 🤝 Contributing

We welcome contributions! If you have ideas for improvements or new features, feel free to:
1. Open an **issue**.
2. Submit a **pull request**.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
