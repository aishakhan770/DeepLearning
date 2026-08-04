# Implement the following functions in Python from scratch. Do not use any library functions. You are allowed to use numpy and matplotlib. Generate 100 equally spaced values between -10 and 10. Call this list as  z. Implement the following functions and its derivative. Use class notes to find the expression for these functions. Use z as input and plot both the function outputs and its derivative outputs.  Upload your code into Github and share it with me.
# Sigmoid
# Tanh
# ReLU (Rectified Linear Unit)
# Leaky ReLU
# Softmax (no need for visualization)

import numpy as np
import matplotlib.pyplot as plt

z=np.linspace(-10,10,100)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    s=sigmoid(z)
    return s*(1-s)

plt.plot(z,sigmoid(z),label='sigmoid')
plt.plot(z,sigmoid_derivative(z),label='sigmoid derivative')
plt.xlabel('z')
plt.ylabel('output')
plt.title('sigmoid function and its derivative')
plt.legend()
plt.grid(True)
plt.show()

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def tanh_derivative(z):
    t=tanh(z)
    return 1- t**2

plt.plot(z,tanh(z),label='tanh')
plt.plot(z,tanh_derivative(z),label='tanh derivative')
plt.xlabel('z')
plt.ylabel('output')
plt.title('tanh function and its derivative')
plt.legend()
plt.grid(True)
plt.show()

def relu(z):
    return np.array([x if x > 0 else 0 for x in z])

def relu_derivative(z):
    return np.array([1 if x >0 else 0 for x in z])

plt.plot(z,relu(z),label='relu')
plt.plot(z,relu_derivative(z),label='relu derivative')
plt.xlabel('z')
plt.ylabel('output')
plt.title('relu function and its derivative')
plt.legend()
plt.grid(True)
plt.show()

def leaky_relu(z,alpha=0.1):
    return np.array([x if x > 0 else alpha* x for x in z])

def leaky_relu_derivative(z,alpha=0.1):
    return np.array([1 if x >0 else alpha for x in z])

plt.plot(z,leaky_relu(z),label='leaky_relu')
plt.plot(z,leaky_relu_derivative(z),label='leaky_relu derivative')
plt.xlabel('z')
plt.ylabel('output')
plt.title('leaky_relu function and its derivative')
plt.legend()
plt.grid(True)
plt.show()

def softmax(z):
    exp_z=np.exp(z)
    return exp_z/np.sum(exp_z)

print(softmax(z))
print("sum",np.sum(softmax(z)))

#Observation
#Sigmoid function
# Min = 0, Max = 1
# Not zero-centred
# For very large or very small inputs, gradient approaches 0.

# Tanh
# Min = -1, Max = 1
# Zero-centred.
# For very large or very small inputs, gradient approaches 0.

# ReLU:
# Min = 0, Max = infinity
# Not zero-centred.
# Gradient = 0 for negative inputs and 1 for positive inputs.

# Leaky ReLU:
# For z = [-10, 10] and alpha = 0.01: Min = -0.1, Max = 10.
# Not strictly zero-centred.
# Gradient = alpha for negative inputs and 1 for positive inputs.

# Softmax:
# Min approaches 0, Max approaches 1.
# Not zero-centred.
# Outputs represent probabilities and their sum is 1.

# Relationship between Sigmoid and Tanh:
# Both are S-shaped functions.
# Tanh is a scaled and shifted version of sigmoid.
# tanh(z) = 2 * sigmoid(2*z) - 1
# Both can suffer from vanishing gradients.