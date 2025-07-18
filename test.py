import os
from PIL import Image
import torch
from torchvision import transforms
from model import CNN  # assuming you moved these into model.py
from labels import LABELS

# Reverse the label dictionary for decoding
INT_TO_LABEL = {v: k for k, v in LABELS.items()}

# Path to your test images
test_folder = "train"

# Define the same transform used in training
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),  # Gives shape [1, 64, 64], perfect for CNN
])

# Load your trained model
model = CNN()
model.load_state_dict(torch.load("CNN_model.pth"))
model.eval()

correct = 0
total = 0

# Go through each image in the test folder
for filename in sorted(os.listdir(test_folder)):
    if filename.endswith(".png"):
        image_path = os.path.join(test_folder, filename)
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        predicted_label = INT_TO_LABEL[predicted_class]

        # Extract expected label from the filename
        expected_label = ''.join([c for c in filename if not c.isdigit()]).replace('.png', '')

        print(f"{filename} → Predicted: {predicted_label}, Expected: {expected_label}")

        if predicted_label == expected_label:
            correct += 1
        total += 1

# Print statistics
print(f"\nCorrect: {correct}/{total} ({100 * correct / total:.2f}%)")