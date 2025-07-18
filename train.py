# train.py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import ShapeDataset
from model import CNN

# Load dataset and dataloader
dataset = ShapeDataset("train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = CNN()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Remove existing model file if it exists
model_path = "CNN_model.pth"
if os.path.exists(model_path):
    os.remove(model_path)
    print(f"Existing {model_path} file removed.")

# Training loop
epochs = 300
for epoch in range(epochs):
    total_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), model_path)
print(f"Model saved as {model_path}")