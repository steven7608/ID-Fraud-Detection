import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\steve\Projects\Real-Time Fraud Detection\fraud-detection\models\driverlicense\resnet50\second_attempt_fix\best_model.pth"
IMG_PATH = r"C:\Users\steve\Desktop\test for demo\combined\combined fraud.png"

NUM_CLASSES = 5
CLASS_NAMES = ["genuine", "copy-move", "face-morph", "face-replace", "combined"]
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL DEFINITION ----------------
def create_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ---------------- IMAGE PREPROCESSING ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0  # for visualization
    return img, img_tensor, img_np

# ---------------- GRAD-CAM++ FUNCTION ----------------
def generate_cam(model, img_tensor, img_np, target_layers):
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor)[0, :]
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return visualization

# ---------------- MAIN ----------------
def main():
    model = create_model()
    img, img_tensor, img_np = preprocess_image(IMG_PATH)

    # Model prediction
    outputs = model(img_tensor)
    pred_class = torch.argmax(outputs, dim=1).item()
    pred_label = CLASS_NAMES[pred_class]
    print(f"Predicted class: {pred_label}")

    # Generate Grad-CAM++ for different layers
    layer3_vis = generate_cam(model, img_tensor, img_np, [model.layer3[-1]])
    layer4_vis = generate_cam(model, img_tensor, img_np, [model.layer4[-1]])

    # Combine both visualizations
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(layer3_vis)
    axes[1].set_title("layer3 - Fine Details")
    axes[1].axis("off")

    axes[2].imshow(layer4_vis)
    axes[2].set_title("layer4 - High-Level Focus")
    axes[2].axis("off")

    plt.suptitle(f"Predicted: {pred_label}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
