#vanishing gradient problem
import numpy as np

layers = 20

# Start with gradient value 1
vanishing = [1.0]
exploding = [1.0]

# Calculate gradient through each layer
for i in range(layers):

    # Gradient becomes smaller
    vanishing.append(vanishing[-1] * 0.5)

    # Gradient becomes larger
    exploding.append(exploding[-1] * 2)

# Print the gradient values
print("Vanishing gradients:")
print(vanishing)

print("\nExploding gradients:")
print(exploding)