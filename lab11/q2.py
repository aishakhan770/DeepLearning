#maxpool from scratch

import numpy as np

# Create a 32 x 32 input image
image = np.random.randn(32, 32)

# Pool size is 2 x 2
pool_size = 2

# Stride is 2
stride = 2

# Output size
output = np.zeros((16, 16))


# Perform max pooling
for i in range(16):

    # Move pooling window down
    for j in range(16):

        # Take a 2 x 2 region
        region = image[
            i*stride:i*stride+pool_size,
            j*stride:j*stride+pool_size
        ]

        # Store the maximum value from the region
        output[i, j] = np.max(region)


# Print results
print("Input image shape:", image.shape)

print("Output shape:", output.shape)

print("MaxPool output:")
print(output[:5, :5])