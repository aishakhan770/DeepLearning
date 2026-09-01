import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

np.random.seed(42) #to generate random numbers

x=np.random.randn(4) #we have 4 input values


print("input x:")
print(x)

#hidden layer 1
#4 inputs have 3 hidden neurons in layer 1
w1=np.random.randn(3,4)

b1=np.random.randn(3) #one bias for 3 neurons
z1=np.zeros(3) #empty vector for z values

#to calculate z for each neuron
for i in range(3):
    for j in range(4): #to calculate weight for each neuron
        z1[i]=z1[i]+w1[i][j]*x[j]
    z1[i]=z1[i]+b1[i] #to add bias term for the neuron

a1=np.zeros(3)

for i in range(3):
    a1[i]=sigmoid(z1[i])

print("\nhidden layer 1")
print("\nWeight matrix:")
print(w1)
print("\nBias matrix:")
print(b1)
print("\nz value:")
print(z1)
print("\nactivation function:")
print(a1)

#hidden layer 2
#the previous layer had 4 inputs and 3 neurons.
#this layer has 3 inputs and 2 neurons

w2=np.random.randn(2,3) #3 inputs to 2 neurons
b2=np.random.randn(2) #each of the 2 neurons has one bias

z2=np.zeros(2)  #vector to store z values

for i in range(2): #to calculate z for each neuron
    for j in range(3): #each neuron receives 3 inputs
        z2[i]=z2[i]+w2[i][j]*a1[j]
    #add the bias term for the neuron
    z2[i]=z2[i]+b2[i]

#apply sigmoid activation
a2=np.zeros(2)

for i in range(2):
    a2[i]=sigmoid(z2[i])
print("\nhidden layer 2")
print("\nWeight matrix:")
print(w2)
print("\nBias matrix:")
print(b2)
print("\nz value:")
print(z2)
print("\nactivation function:")
print(a2)

#output layer

w3=np.random.randn(1,2) #output layer has 1 neuron and previous layer has 2 neurons
b3=np.random.randn(1) #bias to one output neuron
z3=np.zeros(1) #vector to store z
for i in range(1):


    for j in range(2): #the output neuron receives 2 inputs
        z3[i]=z3[i]+w3[i][j]*a2[j]
    #add bias term
    z3[i]=z3[i]+b3[i]

#apply sigmoid function
a3=np.zeros(1)

for i in range(1):
    a3[i]=sigmoid(z3[i])

y_pred=a3[0]

print("\noutput prediction:")
print("\nWeight matrix w3:")
print(w3)
print("\nBias matrix b3:")
print(b3)

print('\nActivation function:')
print(a3)

print("\nFinal output:")
print(y_pred)