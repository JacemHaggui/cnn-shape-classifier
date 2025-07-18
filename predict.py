# predict.py
import torch
from PIL import Image
from model import CNN
from labels import LABELS
from torchvision import transforms

# Define the same preprocessing as in training
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),  # Gives shape [1, 64, 64], perfect for CNN
])

# Load and preprocess image
img_path = "pixil-frame-0-2.png"  # Change this to any test image
image = Image.open(img_path)
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Load model and weights
model = CNN()
model.load_state_dict(torch.load("CNN_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

# Reverse lookup from number to label
label_map = {v: k for k, v in LABELS.items()}
print(f"Predicted shape: {label_map[predicted_class]}")