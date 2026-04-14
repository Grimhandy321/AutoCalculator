import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

CSV_FILE = "../data/data_cleaned.csv"
IMAGE_DIR = "../data/car_images"
MODEL_DIR = "../models"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)
df["listing_id"] = df["listing_id"].astype(str)
df = df.dropna(subset=["brand", "model_extracted", "condition", "image_urls"])

df["image_path"] = df["listing_id"].apply(
    lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg")
)

df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

brand_classes = np.sort(df["brand"].unique())
model_classes = np.sort(df["model_extracted"].unique())
condition_classes = np.sort(df["condition"].unique())

brand_to_idx = {b:i for i,b in enumerate(brand_classes)}
model_to_idx = {m:i for i,m in enumerate(model_classes)}
cond_to_idx = {c:i for i,c in enumerate(condition_classes)}

df["brand_enc"] = df["brand"].map(brand_to_idx)
df["model_enc"] = df["model_extracted"].map(model_to_idx)
df["cond_enc"] = df["condition"].map(cond_to_idx)

brand_model_map = {}

for b in df["brand_enc"].unique():
    brand_model_map[str(b)] = df[df["brand_enc"] == b]["model_enc"].unique().tolist()

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

class CarDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        img = transform(img)

        return (
            img,
            torch.tensor(r["brand_enc"]),
            torch.tensor(r["model_enc"]),
            torch.tensor(r["cond_enc"])
        )

train_loader = DataLoader(CarDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(CarDataset(val_df), batch_size=BATCH_SIZE, shuffle=False)

class HierarchicalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()

        self.shared = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.brand_output = nn.Linear(256, len(brand_classes))
        self.model_output = nn.Linear(256, len(model_classes))
        self.condition_output = nn.Linear(256, len(condition_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)

        return {
            "brand": self.brand_output(x),
            "model": self.model_output(x),
            "condition": self.condition_output(x),
        }

model = HierarchicalModel().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_epoch(loader):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc="Training")

    for imgs, brands, models, conds in loop:
        imgs = imgs.to(DEVICE)
        brands = brands.to(DEVICE)
        models = models.to(DEVICE)
        conds = conds.to(DEVICE)

        optimizer.zero_grad()

        out = model(imgs)

        loss = (
            0.3 * criterion(out["brand"], brands) +
            1.0 * criterion(out["model"], models) +
            0.7 * criterion(out["condition"], conds)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item(), avg=total_loss / (loop.n + 1))

    return total_loss / len(loader)

def validate(loader):
    model.eval()
    correct_b = 0
    correct_m = 0
    correct_c = 0
    total = 0

    with torch.no_grad():
        for imgs, brands, models, conds in loader:
            imgs = imgs.to(DEVICE)
            brands = brands.to(DEVICE)
            models = models.to(DEVICE)
            conds = conds.to(DEVICE)

            out = model(imgs)

            pb = out["brand"].argmax(1)
            pm = out["model"].argmax(1)
            pc = out["condition"].argmax(1)

            correct_b += (pb == brands).sum().item()
            correct_m += (pm == models).sum().item()
            correct_c += (pc == conds).sum().item()
            total += imgs.size(0)

    print(f"Brand acc: {correct_b/total:.3f}")
    print(f"Model acc: {correct_m/total:.3f}")
    print(f"Condition acc: {correct_c/total:.3f}")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_epoch(train_loader)
    validate(val_loader)

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "vision_model_final.pt"))

np.save(os.path.join(MODEL_DIR, "brand_classes.npy"), brand_classes)
np.save(os.path.join(MODEL_DIR, "model_classes.npy"), model_classes)
np.save(os.path.join(MODEL_DIR, "condition_classes.npy"), condition_classes)

print("DONE")