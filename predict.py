import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = models.efficientnet_b0(weights='IMAGENET1K_V1')  # Use updated 'weights' argument
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model.load_state_dict(torch.load('efficientnet_tb_classifier.pth', map_location=device))
model.eval().to(device)

# Transform for single image (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# ✅ Predict function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
        label = "TB" if prob > 0.5 else "Non-TB"
        print(f"Prediction: {label} (Confidence: {prob:.4f})")

# ✅ Ask user for image path
if __name__ == "__main__":
    image_path = input("Please enter the path to the X-ray image: ")
    predict_image(image_path)
