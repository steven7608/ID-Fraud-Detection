#train.py
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision import models
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_DIR = "/Users/vtech/Desktop/f25-16/fraud-detection/data/IDNET/ID Card"
LABELS_FILE = "/Users/vtech/Desktop/f25-16/fraud-detection/labels/IDC_labels.txt"
MODELS_DIR = "/Users/vtech/Desktop/f25-16/fraud-detection/models/idcards/resnet50"
RESULTS_DIR = "/Users/vtech/Desktop/f25-16/fraud-detection/results/idcards/resnet50"

DATA_SUBSET_FRACTION = 1.0
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
FINETUNE_LR = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 3
NUM_CLASSES = 5
IMG_SIZE = 224

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
log_path = os.path.join(RESULTS_DIR, "training_log.txt")
print(f" Training log will be written to: {log_path}")

# ---------------- FOCAL LOSS ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# ---------------- DATASET ----------------
class FraudDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        with open(labels_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                fname = parts[0]
                label = int(parts[1])
                self.samples.append((fname, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Transform failed for {img_path}, got {type(image)}")
        return image, label

# ---------------- TRANSFORMS (correct order) ----------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = T.Compose([
    T.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    T.RandomRotation(15),
    T.RandomPerspective(distortion_scale=0.3, p=0.5),
    T.CenterCrop(IMG_SIZE),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),  # must come before RandomErasing and GaussianBlur
    T.RandomErasing(p=0.3),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ---------------- DATASET SPLIT ----------------
full_dataset = FraudDataset(DATA_DIR, LABELS_FILE, transform=None)
labels = np.array([sample[1] for sample in full_dataset.samples])

if DATA_SUBSET_FRACTION < 1.0:
    subset_indices, _ = train_test_split(
        np.arange(len(full_dataset)),
        test_size=1 - DATA_SUBSET_FRACTION,
        stratify=labels,
        random_state=42
    )
    subset_dataset = Subset(full_dataset, subset_indices)
    print(f"Loaded {len(full_dataset)} samples, using {len(subset_dataset)} for training.")
else:
    subset_dataset = full_dataset
    print(f"Loaded {len(full_dataset)} samples - using entire dataset for training.")

subset_labels = labels[subset_dataset.indices] if isinstance(subset_dataset, Subset) else labels
class_counts = np.bincount(subset_labels, minlength=NUM_CLASSES)
class_weights = [len(subset_labels) / c if c > 0 else 0 for c in class_counts]
alpha = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print(f"Alpha weights: {alpha}")

# ---------------- MODEL ----------------
def create_model(num_classes=NUM_CLASSES):
    model = models.resnet50(weights="IMAGENET1K_V1")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model.to(DEVICE)

# ---------------- TRAIN ONE FOLD ----------------
def train_one_fold(train_idx, val_idx, fold, log_file, alpha):
    print(f"\n--- Fold {fold+1}/5 ---")
    log_file.write(f"\n--- Fold {fold+1}/5 ---\n")
    log_file.flush()

    all_samples = []
    with open(LABELS_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            fname, label = parts[0], int(parts[1])
            all_samples.append((fname, label))

    subset_indices_list = subset_dataset.indices if isinstance(subset_dataset, Subset) else np.arange(len(all_samples))
    subset_samples = [all_samples[i] for i in subset_indices_list]
    train_samples = [subset_samples[i] for i in train_idx]
    val_samples = [subset_samples[i] for i in val_idx]

    class FoldDataset(Dataset):
        def __init__(self, samples, root_dir, transform):
            self.samples = samples
            self.root_dir = root_dir
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            rel_path, label = self.samples[idx]
            full_path = os.path.join(self.root_dir, rel_path)
            image = Image.open(full_path).convert("RGB")
            image = self.transform(image)
            return image, label

    train_dataset = FoldDataset(train_samples, DATA_DIR, train_transforms)
    val_dataset = FoldDataset(val_samples, DATA_DIR, val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    sample_batch, _ = next(iter(train_loader))
    print("Sample batch type:", type(sample_batch), "shape:", getattr(sample_batch, "shape", None))

    model = create_model()
    criterion = FocalLoss(alpha=alpha, gamma=2)

    # --- Phase 1: Train Classifier Head ---
    print("--- Phase 1: Training Classifier Head ---")
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler()

    for epoch in range(2):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Head Epoch {epoch+1}/2"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # --- Phase 2: Fine-tuning ---
    print("\n--- Phase 2: Fine-tuning ---")
    optimizer = optim.AdamW([
        {'params': model.layer3.parameters(), 'lr': FINETUNE_LR},
        {'params': model.layer4.parameters(), 'lr': FINETUNE_LR},
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    best_val_f1 = 0.0
    epochs_no_improve = 0
    temp_model_path = f"temp_best_model_fold_{fold+1}.pth"

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Fine-Tune Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Validation
        model.eval()
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

        current_val_f1 = f1_score(y_true_val, y_pred_val, average='macro', zero_division=0)
        print(f"Fine-Tune Epoch {epoch+1}: Val F1 (Macro)={current_val_f1:.4f}")
        log_file.write(f"Fine-Tune Epoch {epoch+1}: Val F1 (Macro)={current_val_f1:.4f}\n")
        log_file.flush()

        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), temp_model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Final evaluation
    model.load_state_dict(torch.load(temp_model_path))
    os.remove(temp_model_path)
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true, y_pred, y_probs = np.array(y_true), np.array(y_pred), np.array(y_probs)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\nFinal Macro F1 (Fold {fold+1}): {f1_macro:.4f}")
    log_file.write(f"\nFinal Macro F1 (Fold {fold+1}): {f1_macro:.4f}\n")
    log_file.write(classification_report(y_true, y_pred, zero_division=0))
    log_file.flush()

    # --- SAVE GLOBAL BEST MODEL ---
    global best_f1_overall, best_model_path
    if f1_macro > best_f1_overall:
        best_f1_overall = f1_macro
        best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f" New overall best model saved: {best_model_path} (F1={f1_macro:.4f})")
        log_file.write(f" New overall best model saved: {best_model_path} (F1={f1_macro:.4f})\n")
        log_file.flush()

    # Save predictions and confusion matrix
    npz_path = os.path.join(RESULTS_DIR, f"preds_fold_{fold+1}.npz")
    np.savez(npz_path, y_true=y_true, y_pred=y_pred, y_probs=y_probs)
    print(f" Saved fold predictions to {npz_path}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    class_names = ['genuine', 'copy-move', 'face-morph', 'face-replace', 'combined']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold+1}')
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_fold_{fold+1}.png"))
    plt.close()

    return f1_macro

# ---------------- MAIN LOOP ----------------
best_f1_overall, best_model_path = 0.0, None
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_f1_scores = []

with open(log_path, "w") as log_file:
    for fold, (train_idx, val_idx) in enumerate(kf.split(subset_dataset)):
        print(f"\n===== Starting Fold {fold+1}/5 =====")
        log_file.write(f"\n===== Starting Fold {fold+1}/5 =====\n")
        log_file.flush()
        try:
            f1_macro = train_one_fold(train_idx, val_idx, fold, log_file, alpha)
            fold_f1_scores.append(f1_macro)
            log_file.write(f" Completed Fold {fold+1} | Macro F1={f1_macro:.4f}\n")
        except Exception as e:
            print(f" Error during Fold {fold+1}: {e}")
            log_file.write(f" Error during Fold {fold+1}: {e}\n")
            fold_f1_scores.append(0.0)
        finally:
            log_file.flush()

    print("\n===== CROSS-VALIDATION SUMMARY =====")
    log_file.write("\n===== CROSS-VALIDATION SUMMARY =====\n")
    for i, score in enumerate(fold_f1_scores, 1):
        msg = f"Fold {i}: Macro F1 = {score:.4f}"
        print(msg)
        log_file.write(msg + "\n")

    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    summary = f"\nAverage Macro F1: {mean_f1:.4f} ± {std_f1:.4f}\n"
    print(summary)
    log_file.write(summary)
    log_file.flush()
