import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Load the FashionMNIST training dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)


# Load the FashionMNIST test dataset
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# Create DataLoader for training data
train_dataloader = DataLoader(
    training_data,
    batch_size=64,
    shuffle=True
)


# Create DataLoader for test data
test_dataloader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)


# Select GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", device)


# Build the neural network
class NeuralNetwork(nn.Module):

    # Initialize the layers
    def __init__(self):

        # Initialize the parent class
        super().__init__()

        # Flatten converts 28 x 28 image into 784 values
        self.flatten = nn.Flatten()

        # Define the neural network layers
        self.linear_relu_stack = nn.Sequential(

            # 784 input values to 512 neurons
            nn.Linear(28 * 28, 512),

            # Apply ReLU activation
            nn.ReLU(),

            # 512 neurons to 512 neurons
            nn.Linear(512, 512),

            # Apply ReLU activation
            nn.ReLU(),

            # 512 neurons to 10 output classes
            nn.Linear(512, 10)
        )


    # Define the forward pass
    def forward(self, x):

        # Convert image into one-dimensional vector
        x = self.flatten(x)

        # Pass input through all layers
        logits = self.linear_relu_stack(x)

        # Return the output
        return logits


# Create the model
model = NeuralNetwork().to(device)

print(model)


# Define the loss function
loss_fn = nn.CrossEntropyLoss()


# Define the optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-3
)


# Training function
def train(dataloader, model, loss_fn, optimizer):

    # Get total number of samples
    size = len(dataloader.dataset)

    # Put model into training mode
    model.train()

    # Go through each batch
    for batch, (X, y) in enumerate(dataloader):

        # Move input to CPU or GPU
        X = X.to(device)

        # Move labels to CPU or GPU
        y = y.to(device)

        # Forward pass
        pred = model(X)

        # Calculate loss
        loss = loss_fn(pred, y)

        # Clear old gradients
        optimizer.zero_grad()

        # Calculate gradients using backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Print loss after every 100 batches
        if batch % 100 == 0:

            loss_value = loss.item()

            current = batch * len(X)

            print(
                f"loss: {loss_value:>7f} "
                f"[{current:>5d}/{size:>5d}]"
            )


# Testing function
def test(dataloader, model, loss_fn):

    # Get total number of samples
    size = len(dataloader.dataset)

    # Get number of batches
    num_batches = len(dataloader)

    # Put model into evaluation mode
    model.eval()

    # Store total loss
    test_loss = 0

    # Store number of correct predictions
    correct = 0

    # Do not calculate gradients during testing
    with torch.no_grad():

        # Go through all test batches
        for X, y in dataloader:

            # Move input to device
            X = X.to(device)

            # Move labels to device
            y = y.to(device)

            # Get model prediction
            pred = model(X)

            # Calculate test loss
            test_loss += loss_fn(pred, y).item()

            # Get predicted class
            correct += (
                (pred.argmax(1) == y)
                .type(torch.float)
                .sum()
                .item()
            )

    # Calculate average loss
    test_loss /= num_batches

    # Calculate accuracy
    correct /= size

    print(
        f"Test Error: "
        f"Accuracy: {100 * correct:>0.1f}%, "
        f"Avg loss: {test_loss:>8f}"
    )


# Train the model for 5 epochs
epochs = 5

for t in range(epochs):

    print(f"\nEpoch {t + 1}")

    # Perform forward and backward pass
    train(
        train_dataloader,
        model,
        loss_fn,
        optimizer
    )

    # Test the model
    test(
        test_dataloader,
        model,
        loss_fn
    )


# Save the trained model
torch.save(
    model.state_dict(),
    "model.pth"
)

print("\nModel saved!")


# Create a new model with the same architecture
loaded_model = NeuralNetwork().to(device)


# Load the saved weights
loaded_model.load_state_dict(
    torch.load(
        "model.pth",
        weights_only=True
    )
)


# Put loaded model into evaluation mode
loaded_model.eval()

print("\nModel loaded successfully")