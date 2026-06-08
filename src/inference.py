"""
inference.py
-------------------
Classify/Test using a single image using a trained model (post training) 

outputs:
    <predicted class> <cofidence score>
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- CONFIGURATION ---
# 1. Update this to the exact path of your trained model file
MODEL_PATH = "/Users/vtech/Desktop/f25-16/fraud-detection/models/resnet50/driverlicense/best_model.pth"

# 2. Update this to the full path of the image you want to classify
IMAGE_TO_TEST = "/Users/vtech/Desktop/f25-16/fraud-detection/data/IDNET/Drivers License/arizona_dl/positive/generated.photos_0139935.png"

# This mapping must match the one used during training
CLASS_NAMES = {
    0: 'genuine',
    1: 'copy-move_fraud',
    2: 'face-morph_fraud',
    3: 'face-replace_fraud',
    4: 'combined_fraud'
}
NUM_CLASSES = 5
# --- END CONFIGURATION ---

# Set the device (mps for Apple Silicon, cuda for NVIDIA, or cpu)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Model Definition ---
# This function MUST be identical to the one used for training the saved model
def create_model(num_classes=NUM_CLASSES):
    """Creates a ResNet50 model with a custom classifier head."""
    model = models.resnet50(weights=None) # Set weights to None as we are loading our own
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)

# --- Image Transformations ---
# These transformations MUST be identical to the validation transforms used during training
def get_transforms():
    """Returns the image transformations for inference."""
    IMG_SIZE = 160
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# --- Main Prediction Logic ---
def predict(model_path, image_path):
    """
    Loads a trained model and makes a prediction on a single image.
    """
    print(f"Using device: {DEVICE}")

    # Check if the model file and image file exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    # 1. Create model instance and load the saved weights
    print("Loading model...")
    model = create_model()
    try:
        # Load the state dictionary. Use map_location to ensure it loads correctly on CPU/MPS.
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    # Set the model to evaluation mode (important for dropout, batchnorm layers)
    model.eval()

    # 2. Load and transform the input image
    print(f"Loading and processing image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        transform = get_transforms()
        # Transform the image and add a "batch" dimension (B, C, H, W)
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # 3. Make the prediction
    print("Making prediction...")
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model(image_tensor)
        
        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get the top prediction
        top_prob, top_class_index = torch.topk(probabilities, 1)
        
        predicted_class_index = top_class_index.item()
        confidence = top_prob.item()
        
    predicted_class_name = CLASS_NAMES.get(predicted_class_index, "Unknown Class")

    # 4. Print the result
    print("\n--- Prediction Result ---")
    print(f"Predicted Class: {predicted_class_name}")
    print(f"Confidence: {confidence:.2%}")
    print("-------------------------\n")


if __name__ == "__main__":
    predict(MODEL_PATH, IMAGE_TO_TEST)

