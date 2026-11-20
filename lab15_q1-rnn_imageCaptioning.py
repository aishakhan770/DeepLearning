import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. CNN Encoder
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove final classification layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Freeze CNN parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(512, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.fc(features)
        return features


# 2. Vanilla RNN Decoder
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.RNN(
            embed_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)

        # Add image features as the first input
        features = features.unsqueeze(1)

        inputs = torch.cat((features, embeddings), dim=1)

        outputs, _ = self.rnn(inputs)

        outputs = self.fc(outputs)

        return outputs


# Model parameters
embed_size = 256
hidden_size = 512
vocab_size = 5000

encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

print("Vanilla RNN Image Captioning Model Created")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
image = Image.open("sample.jpg").convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Dummy caption (word indices)
captions = torch.randint(0, vocab_size, (1, 5)).to(device)

# Set models to evaluation mode
encoder.eval()
decoder.eval()

# Generate output
with torch.no_grad():
    features = encoder(image)
    outputs = decoder(features, captions)

print("Image Feature Shape:", features.shape)
print("RNN Output Shape:", outputs.shape)

# Get predicted word indices
predictions = outputs.argmax(dim=2)

print("Predicted Word Indices:", predictions)

import matplotlib.pyplot as plt

plt.imshow(Image.open("sample.jpg"))
plt.axis("off")
plt.title("Input Image")
plt.show()

