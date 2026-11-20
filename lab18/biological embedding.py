import torch
import torch.nn as nn

# Protein sequence
sequence = "ACDEFGHIK"

# Amino acid encoding
aa = "ACDEFGHIKLMNPQRSTVWY"
vocab = {a: i+1 for i, a in enumerate(aa)}

# Convert sequence into numbers
x = [vocab[a] for a in sequence]

x = torch.tensor([x])

# Create embedding layer
embed = nn.Embedding(21, 16)

# Generate embeddings
output = embed(x)

print("Protein embedding:")
print(output)

print("Shape:", output.shape)