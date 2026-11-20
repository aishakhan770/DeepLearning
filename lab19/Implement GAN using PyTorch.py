import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Initialize models
G = Generator()
D = Discriminator()

loss_fn = nn.BCELoss()

opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

# Training
for epoch in range(100):

    # Generate fake images
    noise = torch.randn(32, 100)
    fake = G(noise)

    # Create real sample images (random demonstration)
    real = torch.randn(32, 784).tanh()

    # Train discriminator
    real_label = torch.ones(32, 1)
    fake_label = torch.zeros(32, 1)

    loss_D = loss_fn(D(real), real_label) + \
             loss_fn(D(fake.detach()), fake_label)

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # Train generator
    noise = torch.randn(32, 100)
    fake = G(noise)

    loss_G = loss_fn(D(fake), real_label)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    print("Epoch:", epoch+1,
          "Generator Loss:", loss_G.item(),
          "Discriminator Loss:", loss_D.item())