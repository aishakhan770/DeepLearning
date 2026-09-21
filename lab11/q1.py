import numpy as np

# Create a 32 x 32 input image
image = np.random.randn(32, 32)

# Create a 3 x 3 kernel
kernel = np.random.randn(3, 3)

# Output size for stride 1 and no padding
output = np.zeros((30, 30))


# Perform convolution
for i in range(30):

    # Move kernel down the image
    for j in range(30):

        # Move kernel across the image
        region = image[i:i+3, j:j+3]

        # Multiply image region and kernel and add all values
        output[i, j] = np.sum(region * kernel)


# Print results
print("Input image shape:", image.shape)

print("Kernel shape:", kernel.shape)

print("Output shape:", output.shape)

print("\nFirst 5 x 5 values of convolution output:")
print(output[:5, :5])