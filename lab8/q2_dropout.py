import torch


# Create input values
x = torch.tensor([
    [2.0, 4.0, 6.0, 8.0],
    [1.0, 3.0, 5.0, 7.0]
])


# Probability of dropping a neuron
p = 0.5


# Implement dropout from scratch
def dropout(x, p):

    # Generate random values between 0 and 1
    random_values = torch.rand(x.shape)

    # Create mask
    # 0 means dropped
    # 1 means kept
    mask = (random_values > p).float()

    # Apply the mask
    output = x * mask / (1 - p)

    # Return output and mask
    return output, mask


# Apply dropout
output, mask = dropout(x, p)


print("Original input:")
print(x)

print("\nDropout mask:")
print(mask)

print("\nDropped values:")

# Print the original values that were dropped
print(x[mask == 0])

print("\nValues kept:")
print(x[mask == 1])

print("\nOutput after dropout:")
print(output)