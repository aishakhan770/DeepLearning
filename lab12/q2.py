import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Convert CIFAR-10 images to tensors and normalize 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# Load CIFAR-10 training dataset
train_data = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# Divide dataset into batches
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)


# Use GPU if available otherwise CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Define CNN with variable number of convolution layers
class CNN(nn.Module):

    def __init__(self, num_layers):
        super().__init__()

        layers = []

        # CIFAR-10 has 3 input channels
        in_channels = 3

        # Add required number of convolution layers
        for i in range(num_layers):

            # Keep 32 output feature maps
            layers.append(
                nn.Conv2d(
                    in_channels,
                    32,
                    kernel_size=3,
                    padding=1
                )
            )

            # Normalize feature maps
            layers.append(nn.BatchNorm2d(32))

            # Apply ReLU activation
            layers.append(nn.ReLU())

            # Next layer receives 32 channels
            in_channels = 32

        # Store convolution layers
        self.conv = nn.Sequential(*layers)

        # Reduce output to fixed 4 x 4 size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Convert features into 10 class scores
        self.fc = nn.Linear(32 * 4 * 4, 10)


    def forward(self, x):

        # Pass through convolution layers
        x = self.conv(x)

        # Reduce feature map to 4 x 4
        x = self.pool(x)

        # Flatten the feature maps
        x = torch.flatten(x, 1)

        # Pass through output layer
        x = self.fc(x)

        return x


# Train model and return training error
def train_model(num_layers):

    # Create CNN with required number of layers
    model = CNN(num_layers).to(device)

    # Loss function for classification
    criterion = nn.CrossEntropyLoss()

    # Adam updates the weights
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Train for 2 epochs
    for epoch in range(2):

        model.train()

        # Process each batch
        for images, labels in train_loader:

            # Move data to CPU or GPU
            images = images.to(device)
            labels = labels.to(device)

            # Remove old gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(images)

            # Calculate loss
            loss = criterion(output, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()


    # Calculate training accuracy
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Get model predictions
            output = model(images)

            # Select class with highest score
            prediction = output.argmax(dim=1)

            # Count correct predictions
            correct += (prediction == labels).sum().item()

            # Count total images
            total += labels.size(0)


    #  training accuracy
    train_accuracy = 100 * correct / total

    # Training error
    train_error = 100 - train_accuracy

    return train_error


# Different CNN depths to test
layer_numbers = [1, 2, 3, 4, 5, 6]

# Store training error for each network
train_errors = []


# Train CNNs having different numbers of layers
for n in layer_numbers:

    print("Training CNN with", n, "layers")

    error = train_model(n)

    train_errors.append(error)

    print("Training Error:", error, "%")


# Plot number of layers vs training error
plt.plot(
    layer_numbers,
    train_errors,
    marker="o"
)

plt.xlabel("Number of Convolution Layers")
plt.ylabel("Training Error")
plt.title("Training Error vs Number of Layers")

plt.show()