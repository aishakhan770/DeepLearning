import numpy as np

# XOR input
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0], #output
              [1],
              [1],
              [0]])

# Set random seed
np.random.seed(42)

# Initialize weights and biases
W1 = np.random.randn(2,2)
b1 = np.zeros((1,2))

W2 = np.random.randn(2,1)
b2 = np.zeros((1,1))

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Training
for epoch in range(10000):

    # Forward pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    y_pred = sigmoid(z2)

    # Calculate error
    error = y_pred - y

    # Backward pass
    dz2 = error * sigmoid_derivative(z2)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * sigmoid_derivative(z1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update weights
    learning_rate = 1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

# Print final predictions
print("Predictions:")
print(y_pred)

print("\nRounded predictions:")
print(np.round(y_pred))

print("Actual output:")
print(y)