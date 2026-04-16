import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

# CONFIG
CSV_FILE = "../data/bazos_cars_labeled.csv"
IMAGE_DIR = "../data/car_images"
MODEL_DIR = "../models"

IMG_SIZE = 224
BATCH_SIZE = 30
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_FILE, low_memory=False)
df["listing_id"] = df["listing_id"].astype(str)

def find_image_path(listing_id):
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = os.path.join(IMAGE_DIR, f"{listing_id}{ext}")
        if os.path.exists(path):
            return path
    return None

df["image_path"] = df["listing_id"].apply(find_image_path)
df = df.dropna(subset=["image_path", "brand", "model_extracted", "condition"])
df = df.reset_index(drop=True)

print("Rows:", len(df))

# =========================
# FILTER SMALL BRANDS
# =========================
brand_counts = df["brand"].value_counts()
valid_brands = brand_counts[brand_counts >= 5].index
df = df[df["brand"].isin(valid_brands)].reset_index(drop=True)

# =========================
# ENCODING
# =========================
brand_le = LabelEncoder()
model_le = LabelEncoder()
condition_le = LabelEncoder()

df["brand_enc"] = brand_le.fit_transform(df["brand"].astype(str))
df["model_enc"] = model_le.fit_transform(df["model_extracted"].astype(str))
df["condition_enc"] = condition_le.fit_transform(df["condition"].astype(str))

np.save(os.path.join(MODEL_DIR, "brand_classes"), brand_le.classes_)
np.save(os.path.join(MODEL_DIR, "model_classes"), model_le.classes_)
np.save(os.path.join(MODEL_DIR, "condition_classes"), condition_le.classes_)

# =========================
# BRAND-MODEL MASK
# =========================
num_brands = len(brand_le.classes_)
num_models = len(model_le.classes_)

brand_model_mask = np.zeros((num_brands, num_models), dtype=np.float32)

for _, row in df.iterrows():
    brand_model_mask[row["brand_enc"], row["model_enc"]] = 1

brand_model_mask = torch.tensor(brand_model_mask).to(DEVICE)

# =========================
# SPLIT
# =========================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["brand_enc"]
)

# =========================
# DATASET
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

class CarDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        img = transform(img)

        return (
            img,
            torch.tensor(row["brand_enc"], dtype=torch.long),
            torch.tensor(row["model_enc"], dtype=torch.long),
            torch.tensor(row["condition_enc"], dtype=torch.long),
        )

train_loader = DataLoader(CarDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CarDataset(val_df), batch_size=BATCH_SIZE)

# =========================
# MODEL
# =========================
class VisionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()

        self.shared = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.brand_output = nn.Linear(256, num_brands)
        self.model_output = nn.Linear(256, num_models)
        self.condition_output = nn.Linear(256, len(condition_le.classes_))

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)

        return {
            "brand_output": self.brand_output(x),
            "model_output": self.model_output(x),
            "condition_output": self.condition_output(x),
        }

model = VisionModel().to(DEVICE)

# TRAIN SETUP
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# TRACKING
train_losses = []
val_brand_acc = []
val_model_acc = []
val_cond_acc = []

# TRAIN LOOP
def train_epoch(loader):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc="Training", leave=False)

    for imgs, brands, models_gt, conds in loop:
        imgs = imgs.to(DEVICE)
        brands = brands.to(DEVICE)
        models_gt = models_gt.to(DEVICE)
        conds = conds.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(imgs)

        mask = brand_model_mask[brands]
        model_logits = outputs["model_output"].masked_fill(mask == 0, -1e9)

        loss = (
            0.3 * criterion(outputs["brand_output"], brands) +
            1.0 * criterion(model_logits, models_gt) +
            0.7 * criterion(outputs["condition_output"], conds)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / len(loader)

def validate(loader):
    model.eval()

    correct_b = 0
    correct_m = 0
    correct_c = 0
    total = 0

    with torch.no_grad():
        for imgs, brands, models_gt, conds in loader:
            imgs = imgs.to(DEVICE)
            brands = brands.to(DEVICE)
            models_gt = models_gt.to(DEVICE)
            conds = conds.to(DEVICE)

            outputs = model(imgs)

            mask = brand_model_mask[brands]
            model_logits = outputs["model_output"].masked_fill(mask == 0, -1e9)

            pb = outputs["brand_output"].argmax(1)
            pm = model_logits.argmax(1)
            pc = outputs["condition_output"].argmax(1)

            correct_b += (pb == brands).sum().item()
            correct_m += (pm == models_gt).sum().item()
            correct_c += (pc == conds).sum().item()
            total += imgs.size(0)

    return (
        correct_b / total,
        correct_m / total,
        correct_c / total
    )

# RUN TRAINING
for epoch in range(EPOCHS_STAGE1 + EPOCHS_STAGE2):
    print(f"\n=== Epoch {epoch+1} ===")

    loss = train_epoch(train_loader)
    b_acc, m_acc, c_acc = validate(val_loader)

    train_losses.append(loss)
    val_brand_acc.append(b_acc)
    val_model_acc.append(m_acc)
    val_cond_acc.append(c_acc)

    print(f"Loss: {loss:.4f}")
    print(f"Val → Brand: {b_acc:.3f} | Model: {m_acc:.3f} | Cond: {c_acc:.3f}")

# SAVE MODEL
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "vision_model_final.pt"))

# PLOT TRAINING GRAPH
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(val_brand_acc, label="Brand")
plt.plot(val_model_acc, label="Model")
plt.plot(val_cond_acc, label="Condition")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curve.png"))
plt.show()

print("DONE ")