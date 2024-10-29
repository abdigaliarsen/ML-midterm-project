import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time

# Step 1: Define or Load Dataset
# For demonstration, we will create a synthetic dataset with a linear relationship
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Predictor variable
y = 4 + 3 * X + np.random.randn(100, 1)  # Response variable with noise

# Adding a column of ones to X for the intercept term
X_b = np.c_[np.ones((100, 1)), X]

# Step 2: Implement Linear Regression using Normal Equation (NumPy)

# Calculate weights using the normal equation
start_time = time.time()
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
numpy_time = time.time() - start_time

# Extract intercept and slope
intercept_numpy, slope_numpy = theta_best[0][0], theta_best[1][0]

print("NumPy Implementation")
print(f"Intercept: {intercept_numpy}, Slope: {slope_numpy}")
print(f"Computation Time: {numpy_time:.6f} seconds")

# Step 3: Predict Outcomes Using NumPy Model
y_pred_numpy = X_b.dot(theta_best)

# Step 4: Plot the Results
plt.scatter(X, y, color="blue", label="Data Points")
plt.plot(X, y_pred_numpy, color="red", linewidth=2, label="NumPy Regression Line")
plt.xlabel("Predictor (X)")
plt.ylabel("Response (y)")
plt.legend()
plt.title("Linear Regression with NumPy")
plt.show()

# Step 5: Implement Linear Regression using Scikit-learn for Comparison
start_time = time.time()
lin_reg = LinearRegression()
lin_reg.fit(X, y)
sklearn_time = time.time() - start_time

# Scikit-learn results
intercept_sklearn, slope_sklearn = lin_reg.intercept_[0], lin_reg.coef_[0][0]
y_pred_sklearn = lin_reg.predict(X)

print("\nScikit-learn Implementation")
print(f"Intercept: {intercept_sklearn}, Slope: {slope_sklearn}")
print(f"Computation Time: {sklearn_time:.6f} seconds")

# Step 6: Compare Performance Metrics
mse_numpy = mean_squared_error(y, y_pred_numpy)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)

print("\nPerformance Comparison")
print(f"Mean Squared Error (NumPy): {mse_numpy:.6f}")
print(f"Mean Squared Error (Scikit-learn): {mse_sklearn:.6f}")
print(f"Difference in MSE: {abs(mse_numpy - mse_sklearn):.6f}")
print(f"Difference in Computation Time: {abs(numpy_time - sklearn_time):.6f} seconds")

# Additional Plot for Comparison
plt.scatter(X, y, color="blue", label="Data Points")
plt.plot(X, y_pred_numpy, color="red", linestyle="--", label="NumPy Regression Line")
plt.plot(X, y_pred_sklearn, color="green", linestyle="-", label="Scikit-learn Regression Line")
plt.xlabel("Predictor (X)")
plt.ylabel("Response (y)")
plt.legend()
plt.title("Comparison of NumPy and Scikit-learn Regression Lines")
plt.show()
