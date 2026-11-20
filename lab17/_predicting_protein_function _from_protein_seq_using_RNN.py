import torch
import torch.nn as nn

# 1. Sample protein sequences
sequences = ["MKTLLILAV", "ACDEFGHIK",
             "MKWVTFISL", "GGGGAAAAC",
             "MNPQRSTVW"]

# 2. Function labels (3 functions)
Y = torch.tensor([
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0]
], dtype=torch.float32)

# 3. Encode amino acids into numbers
aa = "ACDEFGHIKLMNPQRSTVWY"
vocab = {a: i+1 for i, a in enumerate(aa)}

X = []

for seq in sequences:
    x = [vocab[a] for a in seq]
    x += [0] * (10-len(x))
    X.append(x)

X = torch.tensor(X)

# 4. Build RNN model
class ProteinRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(21, 32, padding_idx=0)
        self.rnn = nn.RNN(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = self.embed(x)        # Embedding
        out, h = self.rnn(x)     # RNN
        out = self.fc(h[-1])     # Predict functions
        return out

# 5. Initialize model
model = ProteinRNN()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Train model
for epoch in range(20):

    optimizer.zero_grad()

    output = model(X)
    loss = criterion(output, Y)

    loss.backward()
    optimizer.step()

    print("Epoch:", epoch+1, "Loss:", loss.item())

# 7. Predict protein functions
model.eval()

with torch.no_grad():
    pred = torch.sigmoid(model(X))

    print("Predicted functions:")
    print((pred > 0.5).int())

#using lstm

import torch
import torch.nn as nn

# 1. Sample protein sequences
sequences = ["MKTLLILAV", "ACDEFGHIK",
             "MKWVTFISL", "GGGGAAAAC",
             "MNPQRSTVW"]

# 2. Function labels (3 functions)
Y = torch.tensor([
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0]
], dtype=torch.float32)

# 3. Convert amino acids into numbers
aa = "ACDEFGHIKLMNPQRSTVWY"
vocab = {a: i+1 for i, a in enumerate(aa)}

X = []
for seq in sequences:
    x = [vocab[a] for a in seq]
    x += [0] * (10-len(x))
    X.append(x)

X = torch.tensor(X)

# 4. Build LSTM model
class ProteinModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(21, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = self.embed(x)       # Embedding
        _, (h, c) = self.lstm(x) # LSTM
        return self.fc(h[-1])   # Prediction

# 5. Initialize model
model = ProteinModel()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Train model
for epoch in range(20):
    optimizer.zero_grad()

    output = model(X)
    loss = criterion(output, Y)

    loss.backward()
    optimizer.step()

    print("Epoch:", epoch+1, "Loss:", loss.item())

# 7. Predict protein functions
model.eval()

with torch.no_grad():
    pred = torch.sigmoid(model(X))
    print("Predicted functions:")
    print((pred > 0.5).int())