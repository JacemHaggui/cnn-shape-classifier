import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from labels import LABELS

class ShapeDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        # List all PNG image files in the given folder
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

        # Define a transform: grayscale, tensor
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),  # Gives shape [1, 64, 64], perfect for CNN
        ])

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get the filename for the given index
        filename = self.image_files[idx]
        image_path = os.path.join(self.folder_path, filename)

        # Open the image and apply the transform
        image = Image.open(image_path)
        image_tensor = self.transform(image)

        # Extract the label from the filename (remove digits and extension)
        label_name = ''.join([c for c in filename if not c.isdigit()]).replace('.png', '')
        label = LABELS[label_name.lower()]

        # Return the image tensor and its label
        return image_tensor, label
    
