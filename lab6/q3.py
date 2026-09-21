import os
import torch
import pandas as pd
import numpy as np

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Create our custom dataset
class FaceLandmarksDataset(Dataset):

    # Initialize the dataset
    def __init__(self, csv_file, root_dir, transform=None):

        # Read the CSV file
        self.landmarks_frame = pd.read_csv(csv_file)

        # Store image folder
        self.root_dir = root_dir

        # Store transformations
        self.transform = transform


    # Return number of samples
    def __len__(self):

        return len(self.landmarks_frame)


    # Return one sample
    def __getitem__(self, idx):

        # Get image name from CSV
        image_name = self.landmarks_frame.iloc[idx, 0]

        # Create complete image path
        image_path = os.path.join(
            self.root_dir,
            image_name
        )

        # Read image
        image = io.imread(image_path)

        # Get landmark coordinates
        landmarks = self.landmarks_frame.iloc[idx, 1:]

        # Convert landmarks to NumPy array
        landmarks = np.array(
            landmarks,
            dtype=float
        )

        # Convert landmarks into x,y pairs
        landmarks = landmarks.reshape(-1, 2)

        # Create sample
        sample = {
            'image': image,
            'landmarks': landmarks
        }

        # Apply transformations
        if self.transform:
            sample = self.transform(sample)

        # Return sample
        return sample


# Resize every image to the same size
class Rescale:

    # Store the required output size
    def __init__(self, output_size):

        self.output_size = output_size


    # Apply rescaling
    def __call__(self, sample):

        # Get image and landmarks
        image = sample['image']
        landmarks = sample['landmarks']

        # Get original height and width
        h, w = image.shape[:2]

        # If output_size is an integer, keep aspect ratio
        if isinstance(self.output_size, int):

            if h > w:
                new_w = self.output_size
                new_h = int(self.output_size * h / w)

            else:
                new_h = self.output_size
                new_w = int(self.output_size * w / h)

        # Resize image
        image = transform.resize(
            image,
            (new_h, new_w)
        )

        # Calculate scaling factors
        scale_x = new_w / w
        scale_y = new_h / h

        # Resize landmark coordinates
        landmarks = landmarks * np.array(
            [scale_x, scale_y]
        )

        # Return resized sample
        return {
            'image': image,
            'landmarks': landmarks
        }


# Crop every image to the same size
class RandomCrop:

    # Store crop size
    def __init__(self, output_size):

        self.output_size = output_size


    # Apply random crop
    def __call__(self, sample):

        # Get image and landmarks
        image = sample['image']
        landmarks = sample['landmarks']

        # Get image dimensions
        h, w = image.shape[:2]

        # Get crop dimensions
        new_h, new_w = self.output_size, self.output_size

        # Choose random top-left corner
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        # Crop image
        image = image[
            top:top + new_h,
            left:left + new_w
        ]

        # Move landmark coordinates
        landmarks = landmarks - np.array(
            [left, top]
        )

        # Return cropped sample
        return {
            'image': image,
            'landmarks': landmarks
        }


# Convert image and landmarks to tensors
class ToTensor:

    # Apply conversion
    def __call__(self, sample):

        # Get image and landmarks
        image = sample['image']
        landmarks = sample['landmarks']

        # Change H,W,C to C,H,W
        image = image.transpose((2, 0, 1))

        # Convert image to tensor
        image = torch.from_numpy(
            image.copy()
        ).float()

        # Convert landmarks to tensor
        landmarks = torch.from_numpy(
            landmarks
        ).float()

        # Return tensors
        return {
            'image': image,
            'landmarks': landmarks
        }


# Apply transformations in order
transform_pipeline = transforms.Compose([
    Rescale(256),
    RandomCrop(224),
    ToTensor()
])


# Create the dataset
dataset = FaceLandmarksDataset(
    csv_file='faces/face_landmarks.csv',
    root_dir='faces/',
    transform=transform_pipeline
)


# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# Print number of samples
print("Number of samples:")
print(len(dataset))


# Get one sample
sample = dataset[0]

# Print image shape
print("\nOne image shape:")
print(sample['image'].shape)

# Print landmark shape
print("\nOne landmark shape:")
print(sample['landmarks'].shape)


# Get one batch
for batch in dataloader:

    # Print batch image shape
    print("\nBatch image shape:")
    print(batch['image'].shape)

    # Print batch landmark shape
    print("\nBatch landmark shape:")
    print(batch['landmarks'].shape)

    # Stop after first batch
    break
#to view
import matplotlib.pyplot as plt

# Get one sample
sample = dataset[0]

# Get image
image = sample['image']

# Get landmarks
landmarks = sample['landmarks']

# Change C,H,W back to H,W,C for matplotlib
image = image.permute(1, 2, 0)

# Display the image
plt.imshow(image)

# Display the 68 landmark points
plt.scatter(
    landmarks[:, 0],
    landmarks[:, 1]
)

# Show the image
plt.show()