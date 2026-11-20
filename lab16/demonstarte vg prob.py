import torch
import torch.nn as nn

# Create input sequence of length 100
x = torch.randn(1, 100, 10)

# Create LSTM and GRU
lstm = nn.LSTM(10, 20, batch_first=True)
gru = nn.GRU(10, 20, batch_first=True)

# Test both models
for name, model in [("LSTM", lstm), ("GRU", gru)]:

    output, _ = model(x)          # Forward propagation

    loss = output[:, -1, :].sum() # Calculate loss

    loss.backward()               # Backpropagation

    # Check gradient
    print(name, "Gradient:",
          model.weight_ih_l0.grad.norm().item())