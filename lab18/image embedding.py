import torch
import torch.nn as nn

# Create CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)       # Extract features
        x = self.relu(x)       # Apply activation
        x = self.pool(x)       # Reduce dimensions
        return x.view(x.size(0), -1)  # Image embedding

# Create model
model = CNN()

# Sample image (1 channel, 28x28)
image = torch.randn(1, 1, 28, 28)

# Generate embedding
embedding = model(image)

print("Image embedding:", embedding)
print("Shape:", embedding.shape)