import torch
import torch.nn as nn
import time

# Create dummy input
x = torch.randn(32, 100, 50)

# Create LSTM and GRU
lstm = nn.LSTM(50, 64, batch_first=True)
gru = nn.GRU(50, 64, batch_first=True)

# Compare both models
for name, model in [("LSTM", lstm), ("GRU", gru)]:

    start = time.time()  # Start timer

    for i in range(100):
        output, _ = model(x)

    end = time.time()    # End timer

    # Calculate number of parameters
    params = sum(p.numel() for p in model.parameters())

    print(name)
    print("Parameters:", params)
    print("Time:", end - start)