#Implement a 1-layer (input - output layer) neural network from scratch for the following dataset.
# This includes implementing forward and backward passes from scratch.
# Print the training loss and plot it over 1000 iterations.

import numpy as np
import matplotlib.pyplot as plt

#sigmoid function

def sigmoid(z):
    return 1/(1+np.exp(-z)) #sigmoid converts the weighted sum into 0 and 1.

#sigmoid derivative

def sigmoid_derivative(z):
    s=sigmoid(z)
    return s*(1-s)

#dataset
x=np.array([
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1],
])

#output values
y=np.array([0,1,1,0])

#initialize weights and bias
np.random.seed(42)

W=np.random.randn(3,1)
b=np.random.randn(1)

learning_rate=0.1 #controls how much we change the weight with each iteration.

loss=[] #store loss values to plot

#training for 1000 iteration
for iteration in range(1000):
    z=x @ W+b

    y_hat=sigmoid(z) #gives prediction for all 4 samples.

    l=np.mean(0.5 *(y_hat - y.reshape(-1,1))**2) #to calculate average loss
    loss.append(l)

#backward pass
    dl_dyhat=y_hat-y.reshape(-1,1) #gradient of loss with respect to prediction

    delta=dl_dyhat*sigmoid_derivative(z) #gradient with respect to z

#gradient of weights
    dw=x.T @ delta /len(x)
#gradient of bias
    db=np.sum(delta,axis=0) /len(x) #add gradients from all four training samples.

#update weights and bias
    W=W - learning_rate*dw #gradient descent
    b=b - learning_rate*db  #update bias
#print loss
    if iteration%100==0:
        print("iteration:",iteration,"loss:",loss)

z=x @ W+b
y_hat=sigmoid(z) #one final forward pass after training

print("final weight:",W)
print("final bias:",b)
print("final output:",y_hat)
print("true values:",y)


#plot training loss
plt.plot(loss)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title("training loss over 1000 iterations")
plt.show()