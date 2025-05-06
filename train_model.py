# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from preprocess import get_dataloaders

# ✅ Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Load Dataloaders
train_loader, val_loader, test_loader, class_names = get_dataloaders()

# ✅ Load Pretrained EfficientNetB0
model = models.efficientnet_b0(pretrained=True)

# Replace final classification head (binary classification)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(device)

# ✅ Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()  # Use with single sigmoid output
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ✅ Training Loop
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()

        acc = 100 * correct / len(train_loader.dataset)
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Accuracy: {acc:.2f}%")

        # Optional: Validate after every epoch
        validate(model, val_loader)

# ✅ Validation Loop
def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%\n")

# ✅ Train the model
train_model(model, train_loader, val_loader, epochs=10)

# ✅ Save the model
torch.save(model.state_dict(), 'efficientnet_tb_classifier.pth')
print("Model saved as efficientnet_tb_classifier.pth")
