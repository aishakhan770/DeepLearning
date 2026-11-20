#Seq2Seq using LSTM
import torch
import torch.nn as nn

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()

        self.embed = nn.Embedding(vocab, hidden)  # Convert words into vectors
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)             # Embedding
        _, (h, c) = self.lstm(x)      # Get hidden and cell states
        return h, c


# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()

        self.embed = nn.Embedding(vocab, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)  # Predict output words

    def forward(self, x, h, c):
        x = self.embed(x)
        out, _ = self.lstm(x, (h, c))  # Use encoder states
        return self.fc(out)


# Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()

        self.encoder = Encoder(vocab, hidden)
        self.decoder = Decoder(vocab, hidden)

    def forward(self, src, trg):
        h, c = self.encoder(src)       # Encode input
        out = self.decoder(trg, h, c)  # Decode output
        return out


# Create model
model = Seq2Seq(100, 64)

# Dummy input and target sequences
src = torch.randint(0, 100, (2, 5))
trg = torch.randint(0, 100, (2, 5))

# Run model
output = model(src, trg)
print(output.shape)

#Seq2Seq using GRU
import torch
import torch.nn as nn

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()

        self.embed = nn.Embedding(vocab, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)       # GRU returns hidden state
        return h


# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()

        self.embed = nn.Embedding(vocab, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, h):
        x = self.embed(x)
        out, _ = self.gru(x, h)
        return self.fc(out)


# Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()

        self.encoder = Encoder(vocab, hidden)
        self.decoder = Decoder(vocab, hidden)

    def forward(self, src, trg):
        h = self.encoder(src)        # Encode input
        out = self.decoder(trg, h)   # Decode output
        return out


# Create model
model = Seq2Seq(100, 64)

# Dummy input and target sequences
src = torch.randint(0, 100, (2, 5))
trg = torch.randint(0, 100, (2, 5))

# Run model
output = model(src, trg)
print(output.shape)


