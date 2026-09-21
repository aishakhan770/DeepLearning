import torch                         # Import PyTorch
import torch.nn as nn                # Import neural network tools


# Create input data
x = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
])


# Create correct output values
y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0]
])


# Create a neural network with 2 inputs and 1 output
model = nn.Linear(2, 1)


# Define Mean Squared Error as the loss function
loss_fn = nn.MSELoss()


# SGD updates weights using the current gradient
optimizer_sgd = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# Momentum remembers previous updates to make learning faster and smoother
optimizer_momentum = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)


# AdaGrad gives different learning rates to different parameters
optimizer_adagrad = torch.optim.Adagrad(
    model.parameters(),
    lr=0.01
)


# Adam combines momentum with adaptive learning rates
optimizer_adam = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# Function to perform one training step
def train_step(optimizer):

    # Remove gradients from the previous step
    optimizer.zero_grad()

    # Calculate prediction using forward pass
    y_pred = model(x)

    # Calculate the error between prediction and target
    loss = loss_fn(y_pred, y)

    # Calculate gradients using backpropagation
    loss.backward()

    # Update weights using the selected optimizer
    optimizer.step()

    # Return the loss value
    return loss.item()


# Train using SGD
loss = train_step(optimizer_sgd)

print("SGD loss:")
print(loss)


# Train using Momentum
loss = train_step(optimizer_momentum)

print("\nMomentum loss:")
print(loss)


# Train using AdaGrad
loss = train_step(optimizer_adagrad)

print("\nAdaGrad loss:")
print(loss)


# Train using Adam
loss = train_step(optimizer_adam)

print("\nAdam loss:")
print(loss)