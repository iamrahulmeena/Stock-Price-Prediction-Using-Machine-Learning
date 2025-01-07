import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt

# Function to load the model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    # Predict on the test data
    y_pred = model.predict(X_test)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate R^2 Score
    r2 = r2_score(y_test, y_pred)
    
    # Calculate the percentage of error reduction
    initial_loss = np.mean(np.abs(y_test - np.mean(y_test)))  # Initial loss (mean error)
    final_loss = mae  # Final loss (after prediction)
    reduction_percentage = ((initial_loss - final_loss) / initial_loss) * 100
    
    # Calculate accuracy as R^2 percentage
    accuracy = r2 * 100  # R^2 Score in percentage
    
    # Display the results
    print("Evaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae:.5f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.5f}")
    print(f"R^2 Score: {r2:.5f}")
    print(f"Accuracy (R^2 * 100): {accuracy:.2f}%")
    print(f"Initial Loss (before prediction): {initial_loss:.5f}")
    print(f"Final Loss (after prediction): {final_loss:.5f}")
    print(f"Error Reduction: {reduction_percentage:.2f}%")
    
    return y_pred, reduction_percentage, accuracy

# Function to plot predictions vs actual
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Prices", color='blue')
    plt.plot(y_pred, label="Predicted Prices", color='red')
    plt.legend()
    plt.title("Stock Price Prediction: Actual vs Predicted")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.show()

# Main function to load and evaluate the model
def main():
    model_path = "path_to_your_model.pkl"  # Replace with the path to your trained model
    X_test = np.load('X_test.npy')  # Replace with your test data
    y_test = np.load('y_test.npy')  # Replace with your true labels/stock prices

    # Load the trained model
    model = load_model(model_path)

    # Evaluate the model and get predictions
    y_pred, reduction_percentage, accuracy = evaluate_model(model, X_test, y_test)
    
    # Plot the predictions vs actual
    plot_predictions(y_test, y_pred)

# Run the main function
if __name__ == "__main__":
    main()
