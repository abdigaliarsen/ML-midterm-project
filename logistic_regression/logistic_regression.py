import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# Step 1: Load and preprocess dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Add intercept column of ones to X
X = np.c_[np.ones((X.shape[0], 1)), X]  # Intercept term
y = y.reshape(-1, 1)  # Reshape for compatibility

# Step 2: Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 3: Define the gradient of the in-sample error function
def compute_gradient(X, y, w):
    N = len(y)
    predictions = sigmoid(X.dot(w))
    errors = predictions - y
    gradient = (1 / N) * X.T.dot(errors)
    return gradient

# Step 4: Implement gradient descent
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    w = np.zeros((X.shape[1], 1))  # Initialize weights
    for i in range(iterations):
        gradient = compute_gradient(X, y, w)
        w -= learning_rate * gradient
    return w

# Fit the model using gradient descent
start_time = time.time()
w_optimal = gradient_descent(X, y, learning_rate=0.01, iterations=5000)
numpy_time = time.time() - start_time

# Step 5: Predict class probabilities and set decision boundary
def predict(X, w):
    return sigmoid(X.dot(w))

y_pred_prob = predict(X, w_optimal)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Step 6: Evaluate performance of the NumPy implementation
accuracy_numpy = accuracy_score(y, y_pred)
print("NumPy Implementation")
print(f"Accuracy: {accuracy_numpy * 100:.2f}%")
print(f"Computation Time: {numpy_time:.6f} seconds")

# Plot decision boundary (for a 2D representation - here for illustration only)
plt.figure(figsize=(8, 6))
plt.plot(range(len(y)), y, 'bo', label='Actual')
plt.plot(range(len(y)), y_pred_prob, 'r.', label='Predicted Probability')
plt.xlabel("Sample index")
plt.ylabel("Class probability")
plt.title("Predicted probabilities (NumPy)")
plt.legend()
plt.show()

# Step 7: Compare with Scikit-learn implementation
start_time = time.time()
sklearn_model = LogisticRegression(max_iter=5000)
sklearn_model.fit(X[:, 1:], y.ravel())  # Skip intercept term in Scikit-learn
y_pred_sklearn = sklearn_model.predict(X[:, 1:])
sklearn_time = time.time() - start_time

# Evaluate Scikit-learn performance
accuracy_sklearn = accuracy_score(y, y_pred_sklearn)
print("\nScikit-learn Implementation")
print(f"Accuracy: {accuracy_sklearn * 100:.2f}%")
print(f"Computation Time: {sklearn_time:.6f} seconds")

# Final performance comparison
print("\nPerformance Comparison")
print(f"Accuracy Difference: {abs(accuracy_numpy - accuracy_sklearn) * 100:.2f}%")
print(f"Time Difference: {abs(numpy_time - sklearn_time):.6f} seconds")


