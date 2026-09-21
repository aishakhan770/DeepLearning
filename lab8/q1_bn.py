import torch

eps=1e-5 #small value to avoid division by zero

x=torch.tensor([
 [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0],
    [3.0, 6.0, 9.0],
    [4.0, 8.0, 12.0]
])

# Gamma controls scaling
gamma = torch.tensor([1.0, 1.0, 1.0])

# Beta controls shifting
beta = torch.tensor([0.0, 0.0, 0.0])

#batch normalization
def batch_normalization(x,gamma,beta):
    # Calculate mean for each feature
    mean = x.mean(dim=0) #dim=0 means column wise

    # Calculate variance for each feature
    variance = x.var(dim=0, unbiased=False)

    # Normalize the input
    x_hat = (x - mean) / torch.sqrt(variance + eps)

    # Scale and shift the normalized values
    y = gamma * x_hat + beta

    # Return final output
    return y

#calculate batchnorm output
y_hat = batch_normalization(x,gamma,beta)

print("input:",x)
print("output:",y_hat)

#layer normalization
def layer_norm(x,gamma,beta):
    mean = x.mean(dim=1,keepdim=True) #calculates mean for each sample
    # Calculate variance for each sample
    variance = x.var(
        dim=1, #row wise
        keepdim=True,
        unbiased=False
    )

    # Normalize each sample
    x_hat = (x - mean) / torch.sqrt(variance + eps)

    # Scale and shift the normalized values
    y = gamma * x_hat + beta

    # Return final output
    return y

# Calculate LayerNorm output
y_layer = layer_norm(x, gamma, beta)

print("\nLayer Normalization output:")
print(y_layer)