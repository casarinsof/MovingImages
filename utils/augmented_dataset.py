import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AugmentedDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Read the txt file to get the image paths and corresponding labels
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                path,  _, label = line.strip().split()
                for i in range(20):
                    new_path = os.path.join(path, f'frames_{i + 1:06d}.png')
                    self.samples.append((new_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage:
#train_folder = './data/moving_cifar-10/train'
#train_txt_file = '/home/sofiacasarin/PycharmProjects/MovingImages/data/moving_cifar-10/train_videofolder.txt'

# Define the transforms you want to apply to the images (e.g., resize, normalize, etc.)


# Create the dataset instance
#dataset = AugmentedDataset(root_dir=train_folder, txt_file=train_txt_file, transform=transform)

# Create a DataLoader to load the data in batches
#batch_size = 32
#train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
