import numpy as np
import matplotlib.pyplot as plt

layers = 20

# Start with gradient value 1
vanishing = [1.0]
exploding = [1.0]

# Calculate gradient for each layer
for i in range(layers):

    # Gradient decreases at every layer
    vanishing.append(vanishing[-1] * 0.5)

    # Gradient increases at every layer
    exploding.append(exploding[-1] * 2)

# Plot vanishing gradient
plt.plot(vanishing, label="Vanishing Gradient")

# Plot exploding gradient
plt.plot(exploding, label="Exploding Gradient")

plt.xlabel("Number of Layers")
plt.ylabel("Gradient Value")


plt.title("Vanishing and Exploding Gradients")

plt.legend()

plt.show()