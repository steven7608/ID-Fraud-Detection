"""
classical_models.py
-------------------
Trains and evaluates Random Forest and HGBDT classifiers on CNN embeddings extracted from extract_features.py

Outputs:
  - rf_f1_vs_estimators.png
  - hgbdt_f1_vs_estimators.png
  - classical_results.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# ---------------- CONFIGURATION ----------------
RESULTS_DIR = "/Users/vtech/Desktop/f25-16/fraud-detection/results/resnet50/driverslicense"
CLASS_NAMES = ['genuine', 'copy-move', 'face-morph', 'face-replace', 'combined']
RANDOM_STATE = 42
# ------------------------------------------------

features = np.load(os.path.join(RESULTS_DIR, "features.npy"))
labels = np.load(os.path.join(RESULTS_DIR, "labels.npy"))
print(f"Loaded features: {features.shape}, labels: {labels.shape}")

# Split into training/testing (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
)

# ---------------- RANDOM FOREST ----------------
rf_estimators = [10, 50, 100, 200, 500]
rf_results = []

for n in rf_estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    rf_results.append((n, f1))
    print(f"RF (estimators={n}) → Macro F1 = {f1:.4f}")

# Plot RF F1 curve
plt.figure()
plt.plot([r[0] for r in rf_results], [r[1] for r in rf_results], marker='o', color='b')
plt.title("Random Forest: F1 Score vs Number of Estimators")
plt.xlabel("Number of Estimators")
plt.ylabel("Macro F1 Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "rf_f1_vs_estimators.png"))

# ---------------- HGBDT ----------------
hgbdt_estimators = [50, 100, 200, 400]
hgbdt_results = []

for n in hgbdt_estimators:
    hgbdt = HistGradientBoostingClassifier(max_iter=n, random_state=RANDOM_STATE)
    hgbdt.fit(X_train, y_train)
    y_pred = hgbdt.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    hgbdt_results.append((n, f1))
    print(f"HGBDT (estimators={n}) → Macro F1 = {f1:.4f}")

# Plot HGBDT F1 curve
plt.figure()
plt.plot([r[0] for r in hgbdt_results], [r[1] for r in hgbdt_results], marker='o', color='r')
plt.title("HGBDT: F1 Score vs Number of Estimators")
plt.xlabel("Number of Iterations")
plt.ylabel("Macro F1 Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "hgbdt_f1_vs_estimators.png"))

# ---------------- SAVE RESULTS ----------------
df_rf = pd.DataFrame(rf_results, columns=["Estimators", "Macro_F1_RF"])
df_hgbdt = pd.DataFrame(hgbdt_results, columns=["Estimators", "Macro_F1_HGBDT"])
df_combined = pd.concat([df_rf, df_hgbdt], axis=1)
df_combined.to_csv(os.path.join(RESULTS_DIR, "classical_results.csv"), index=False)

print("\n=== Classification Report (Best RF) ===")
best_rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES, zero_division=0))

print("\n All results saved to:", RESULTS_DIR)

