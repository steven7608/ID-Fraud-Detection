"""
extract_features.py
-------------------
Extract CNN feature embeddings from trained ResNet50 for downstream classical ML models used for benchamrking test using RF and HGDBT
(Random Forest and HGBDT).

Saves:
  - features.npy
  - labels.npy
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIGURATION ----------------
MODEL_PATH = "/Users/vtech/Desktop/f25-16/fraud-detection/models/resnet50/driverlicense/best_model.pth"
DATA_DIR = "/Users/vtech/Desktop/f25-16/fraud-detection/data/IDNET/Drivers License"
LABELS_FILE = "/Users/vtech/Desktop/f25-16/fraud-detection/DL_labels.txt"
SAVE_DIR = "/Users/vtech/Desktop/f25-16/fraud-detection/results/resnet50/driverslicense"
IMG_SIZE = 160
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# ------------------------------------------------

# Same normalization as training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ---------------- MODEL DEFINITION ----------------
def create_feature_extractor():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Identity()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    return model



# ---------------- DATASET LOADER ----------------
def load_dataset(labels_file, data_dir):
    samples = []
    with open(labels_file, "r") as f:
        for line in f:
            path, label = line.strip().split()
            samples.append((os.path.join(data_dir, path), int(label)))
    return samples


def extract_features():
    print(f"Using device: {DEVICE}")
    model = create_feature_extractor()
    samples = load_dataset(LABELS_FILE, DATA_DIR)

    features, labels = [], []
    with torch.no_grad():
        for img_path, label in tqdm(samples, desc="Extracting CNN Features"):
            try:
                image = Image.open(img_path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(DEVICE)
                feat = model(tensor).squeeze().cpu().numpy()
                features.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    features = np.array(features)
    labels = np.array(labels)
    np.save(os.path.join(SAVE_DIR, "features.npy"), features)
    np.save(os.path.join(SAVE_DIR, "labels.npy"), labels)

    print(f" Saved features.npy and labels.npy to {SAVE_DIR}")
    print(f"Feature matrix shape: {features.shape}, Labels shape: {labels.shape}")


if __name__ == "__main__":
    extract_features()

