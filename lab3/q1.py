import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    s=sigmoid(z)
    return s*(1-s)

np.random.seed(42) #to keep same random values every time

#forward pass
x=np.random.randn(4) #4 input values
y=1 #prediction of y(output)

#hidden layer 1
w1 = np.random.randn(3,4)
b1 = np.random.randn(3)

z1=w1 @ x+b1
a1=sigmoid(z1)


#hidden layer 2
w2=np.random.randn(2,3) #3 inputs -2 neurons
b2=np.random.randn(2) #adding bias term to 2 neurons

z2=w2 @ a1 + b2
a2=sigmoid(z2)

#output layer
w3=np.random.randn(1,2) #2 inputs in 1 neuron

b3=np.random.randn(1) #add one bias term

z3=w3 @ a2+b3
a3=sigmoid(z3)

y_hat=a3[0]

# Mean squared error for one sample
loss = 0.5 * (y_hat - y) ** 2

print("Prediction y_hat:")
print(y_hat)

print("\nLoss:")
print(loss)

#backward pass
dl_dy_hat=y_hat - y
#dl/dz3
# Chain rule:
# dL/dz3 = dL/dy_hat * dy_hat/dz3
delta3 = dl_dy_hat * sigmoid_derivative(z3)
# shape (1,2)
#
# -1 tells Numpy to automatically calculate
# the number of columns.
# Gradient of b3
dW3 = delta3 * a2.reshape(1, -1) #reshaping to make the matrix dimension match

db3=delta3 #derivative of b3 is 1



dL_da2 = w3.T @ delta3 # Gradient flowing backward to a2

delta2 = dL_da2 * sigmoid_derivative(z2) # Gradient with respect to z2
dW2 = delta2.reshape(-1, 1) @ a1.reshape(1, -1) # Gradient of W2
db2 = delta2 # Gradient of b2

#hidden layer 1
#gradient flowing backward to a1

#w2 has shape (2,3)
#w2.T has shape (3,2)
#delta2 has shape(2,)
# Result has shape (3,)
dL_da1=w2.T @ delta2

#GRADIENT WITH RESPECT TO Z1

delta1=dL_da1 *sigmoid_derivative(z1)

# Gradient of W1
# W1 has shape (3,4)
# delta1 has shape (3,)
# x has shape (4,)
# Reshape:
# This matches the shape of W1.
dW1 = delta1.reshape(-1, 1) @ x.reshape(1, -1)

# Gradient of bias b1
db1 = delta1
print("backward pass")
print("\nOutput layer")
print("\nGradient of neuron")
print(delta3)


print("\nGradient of W3:")
print(dW3)

print("\nGradient of b3:")
print(db3)

#Hidden layer 2

print("\nHidden Layer 2")

print("\nGradient of neurons (delta2):")
print(delta2)

print("\nGradient of W2:")
print(dW2)

print("\nGradient of b2:")
print(db2)

# Hidden Layer 1
print("\nHidden Layer 1")
print("\nGradient of neurons (delta1):")
print(delta1)
print("\nGradient of W1:")
print(dW1)
print("\nGradient of b1:")
print(db1)
