import numpy as np

def relu(z):
    return np.maximum(0, z) #if z is positive,return z . if z is negative,return 0

np.random.seed(42) #to get random values everytime

x=np.random.randn(4) #we have 4 input values

print("input x :")
print(x)

W=np.random.randn(1,4) #4 inputs and 1 output neuron
b=np.random.randn(1)
z = W @ x + b #performs matrix multiplication
a=relu(z)

y_hat=a[0]
print("\nWeight matrix W:")
print(W)

print("\nBias b:")
print(b)

print("\nz value:")
print(z)

print("\nActivation value:")
print(a)

print("\nFinal prediction y_hat:")
print(y_hat)

print("\nNetwork 2")

x=np.random.randn(4)
print("input x :")
print(x)

w1=np.random.randn(3,4) #3 neurons receiving 4 inputs
b1=np.random.randn(3) #one bias for each neuron

z1=w1 @ x +b1 # to calculate weighted sum
a1=relu(z1) #relu is the activation function

print("\nHidden layer 1")

print("weight matrix w1:")
print(w1)
print("\nBias layer 1:")
print(b1)
print("\nz value:")
print(z1)
print("\nActivation value:")
print(a1)

#hidden layer 2
w2=np.random.randn(2,3) #2 neurons receiving 3 inputs
b2=np.random.randn(2)

z2=w2 @ a1 +b2
a2=relu(z2)

print("\nHidden layer 2")
print("weight matrix w2:")
print(w2)
print("\nBias layer 2:")
print(b2)
print("\nz value:")
print(z2)
print("\nActivation value:")
print(a2)

#output layer
print("\nFinal prediction y_hat:")

w3=np.random.randn(1,2) #one neuron having 2 inputs
b3=np.random.randn(1)

z3=w3 @ a2 +b3
a3=relu(z3)
print("\nOutput layer")
print("weight matrix w3:")
print(w3)
print("\nBias layer 3:")
print(b3)
print("\nz value:")
print(z3)
print("\nActivation value:")
print(a3)
print("\nFinal prediction y_hat:")
print("y_hat value:")
print(a3[0])




