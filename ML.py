# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data (House size vs Price)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # House size (in 1000 sq ft)
y = 4 + 3 * X + np.random.randn(100, 1)  # Price formula with noise

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Model Coefficient:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Squared Error:", mse)

# Plot results
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.legend()
plt.show()
