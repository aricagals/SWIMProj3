from sklearn.pipeline import Pipeline
from swimnetworks import Dense, Linear
import numpy as np
import matplotlib.pyplot as plt

steps = [
    ("dense", Dense(layer_width=512, activation="tanh",
                     parameter_sampler="tanh",
                     random_seed=42)),
    ("linear", Linear(regularization_scale=1e-10))
]
model = Pipeline(steps)

def generate_simple_dataset(n_samples=100, seed=42):
    np.random.seed(seed)
    
    # Input: uniform samples in [-π, π] for both x and y
    X = np.random.uniform(-np.pi, np.pi, size=(n_samples, 2))
    
    # Output: f(x, y) = sin(x) + cos(y)
    Y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    Y = Y.reshape(-1, 1)  # shape (n_samples, 1)
    
    return X, Y

X_train, y_train = generate_simple_dataset(n_samples=1000, seed=42)
X_test, y_test = generate_simple_dataset(n_samples=100, seed=42)

model.fit(X_train, y_train)
test_predictions = model.transform(X_test)

# Plot predictions y[1] and test y[1], and differences
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_predictions, cmap='viridis')
plt.title("Predictions")
plt.colorbar(label='Predicted Value')

plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
plt.title("True Values")
plt.colorbar(label='True Value')

plt.subplot(1, 3, 3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_predictions - y_test, cmap='coolwarm')
plt.title("Difference")
plt.colorbar(label='Difference')

plt.suptitle("Model Predictions vs True Values")
plt.show()




