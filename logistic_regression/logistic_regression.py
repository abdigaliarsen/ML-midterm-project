import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
from scipy.optimize import minimize

# Step 1: Load and Preprocess the Dataset
def load_and_preprocess_data():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split into train and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize the features to have zero mean and unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Add intercept term (column of ones) for manual implementations
    X_train_scaled = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
    X_test_scaled = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

    # Keep labels as {0, 1} for compatibility
    y_train_converted = y_train.reshape(-1)  # Ensure 1D array
    y_test_converted = y_test.reshape(-1)    # Ensure 1D array

    return X_train_scaled, X_test_scaled, y_train_converted, y_test_converted

# Step 2: Define the Sigmoid Function with Numerical Stability
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))

# Step 3: Define Logistic Regression with Gradient Descent
class LogisticRegressionWithGradientDescent:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):
            # Linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid to get predictions
            predictions = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)  # Predict {0,1}

# Step 4: Implementing Logistic Regression with SciPy Optimization
class LogisticRegressionWithSciPy:
    def __init__(self):
        self.weights = None

    def cost_function(self, weights, X, y):
        """
        Compute the cost function for logistic regression with numerical stability.
        """
        linear_model = np.dot(X, weights)
        predictions = sigmoid(linear_model)
        # To prevent log(0), clip predictions
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return cost

    def gradient(self, weights, X, y):
        """
        Compute the gradient of the cost function.
        """
        linear_model = np.dot(X, weights)
        predictions = sigmoid(linear_model)
        gradient = np.dot(X.T, (predictions - y)) / len(y)
        return gradient

    def fit(self, X, y):
        n_features = X.shape[1]
        initial_weights = np.zeros(n_features)

        # Minimize the cost function using BFGS algorithm
        result = minimize(
            fun=self.cost_function,
            x0=initial_weights,
            args=(X, y),
            method='BFGS',
            jac=self.gradient,
            options={'maxiter': 1000, 'disp': False}
        )

        self.weights = result.x

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights)
        return sigmoid(linear_model)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)  # Predict {0,1}

# Step 5: Visualization
def plot_predictions(y_true, y_pred_manual_gd, y_pred_manual_scipy, y_pred_sklearn):
    plt.figure(figsize=(18, 6))
    
    # Plot Manual Gradient Descent Predictions
    plt.subplot(1, 3, 1)
    plt.scatter(range(len(y_true)), y_true, color='blue', label='Actual', alpha=0.6)
    plt.scatter(range(len(y_pred_manual_gd)), y_pred_manual_gd, color='red', label='GD Predicted', alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.title("Gradient Descent Predictions")
    plt.legend()
    
    # Plot SciPy Optimization Predictions
    plt.subplot(1, 3, 2)
    plt.scatter(range(len(y_true)), y_true, color='blue', label='Actual', alpha=0.6)
    plt.scatter(range(len(y_pred_manual_scipy)), y_pred_manual_scipy, color='green', label='SciPy Predicted', alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.title("SciPy Optimization Predictions")
    plt.legend()
    
    # Plot Scikit-learn Predictions
    plt.subplot(1, 3, 3)
    plt.scatter(range(len(y_true)), y_true, color='blue', label='Actual', alpha=0.6)
    plt.scatter(range(len(y_pred_sklearn)), y_pred_sklearn, color='purple', label='Sklearn Predicted', alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.title("Scikit-learn Predictions")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    X_train, X_test, y_train_manual, y_test_manual = load_and_preprocess_data()

    # Manual Gradient Descent Implementation
    log_reg_gd = LogisticRegressionWithGradientDescent(learning_rate=0.01, iterations=1000)
    start_time = time.time()
    log_reg_gd.fit(X_train, y_train_manual)
    gd_time = time.time() - start_time

    predictions_gd = log_reg_gd.predict(X_test)
    accuracy_gd = accuracy_score(y_test_manual, predictions_gd)
    print(f"Gradient Descent Accuracy: {accuracy_gd:.4f}, Time: {gd_time:.4f} seconds")

    # Manual SciPy Optimization Implementation
    log_reg_scipy = LogisticRegressionWithSciPy()
    start_time = time.time()
    log_reg_scipy.fit(X_train, y_train_manual)
    scipy_time = time.time() - start_time

    predictions_scipy = log_reg_scipy.predict(X_test)
    accuracy_scipy = accuracy_score(y_test_manual, predictions_scipy)
    print(f"SciPy Optimization Accuracy: {accuracy_scipy:.4f}, Time: {scipy_time:.4f} seconds")

    # Scikit-learn Implementation
    start_time = time.time()
    sklearn_model = SklearnLogisticRegression(max_iter=1000)
    sklearn_model.fit(X_train[:, 1:], y_train_manual)  # Scikit-learn handles intercept
    sklearn_time = time.time() - start_time

    predictions_sklearn = sklearn_model.predict(X_test[:, 1:])
    accuracy_sklearn = accuracy_score(y_test_manual, predictions_sklearn)
    print(f"Scikit-learn Accuracy: {accuracy_sklearn:.4f}, Time: {sklearn_time:.4f} seconds\n")

    # Comparison of Implementations
    print("Comparison of Implementations:")
    print(f"Gradient Descent – Accuracy: {accuracy_gd:.4f}, Time: {gd_time:.4f} seconds")
    print(f"SciPy Optimization – Accuracy: {accuracy_scipy:.4f}, Time: {scipy_time:.4f} seconds")
    print(f"Scikit-learn – Accuracy: {accuracy_sklearn:.4f}, Time: {sklearn_time:.4f} seconds")

    # Visualization
    plot_predictions(y_test_manual, predictions_gd, predictions_scipy, predictions_sklearn)

if __name__ == "__main__":
    main()
