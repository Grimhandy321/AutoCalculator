import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms, models

CSV_FILE = "../data/bazos_cars_labeled.csv"
IMAGE_DIR = "../data/car_images"
MODEL_DIR = "../models"
OUTPUT_FILE = "../data/vision_predictions.csv"

IMG_SIZE = 224
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(CSV_FILE)
df["listing_id"] = df["listing_id"].astype(str)

brand_classes = np.load(os.path.join(MODEL_DIR, "brand_classes.npy"), allow_pickle=True)
model_classes = np.load(os.path.join(MODEL_DIR, "model_classes.npy"), allow_pickle=True)
condition_classes = np.load(os.path.join(MODEL_DIR, "condition_classes.npy"), allow_pickle=True)

def find_image_path(listing_id):
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = os.path.join(IMAGE_DIR, f"{listing_id}{ext}")
        if os.path.exists(path):
            return path
    return None

df["image_path"] = df["listing_id"].apply(find_image_path)
df = df.dropna(subset=["image_path"]).reset_index(drop=True)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class VisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = torch.nn.Identity()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(1280, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4)
        )
        self.brand_output = torch.nn.Linear(256, len(brand_classes))
        self.model_output = torch.nn.Linear(256, len(model_classes))
        self.condition_output = torch.nn.Linear(256, len(condition_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)
        return {
            "brand_output": self.brand_output(x),
            "model_output": self.model_output(x),
            "condition_output": self.condition_output(x),
        }

model = VisionModel().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "vision_model_final.pt"), map_location=DEVICE))
model.eval()

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img)
    return img

rows = []

with torch.no_grad():
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]

        images = []
        ids = []

        for _, row in batch.iterrows():
            try:
                images.append(load_image(row["image_path"]))
                ids.append(row["listing_id"])
            except:
                continue

        if not images:
            continue

        images = torch.stack(images).to(DEVICE)

        outputs = model(images)

        brand_pred = torch.softmax(outputs["brand_output"], dim=1).cpu().numpy()
        model_pred = torch.softmax(outputs["model_output"], dim=1).cpu().numpy()
        condition_pred = torch.softmax(outputs["condition_output"], dim=1).cpu().numpy()

        for j, listing_id in enumerate(ids):
            rows.append({
                "listing_id": listing_id,
                "pred_brand": brand_classes[np.argmax(brand_pred[j])],
                "pred_model": model_classes[np.argmax(model_pred[j])],
                "pred_condition": condition_classes[np.argmax(condition_pred[j])],
                "brand_conf": float(np.max(brand_pred[j])),
                "model_conf": float(np.max(model_pred[j])),
                "condition_conf": float(np.max(condition_pred[j]))
            })

        print(f"Processed {min(i + BATCH_SIZE, len(df))}/{len(df)}")

pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Saved predictions: {OUTPUT_FILE}")