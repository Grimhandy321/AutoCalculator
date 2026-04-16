import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from models.loaders import load_model

CSV_FILE = "../data/bazos_cars_labeled.csv"
IMAGE_DIR = "../data/car_images"
MODEL_DIR = "../models"
OUTPUT_FILE = "../data/vision_predictions.csv"

IMG_SIZE = 224
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(CSV_FILE, low_memory=False)
df["listing_id"] = df["listing_id"].astype(str)

model, brand_classes, model_classes, condition_classes = load_model(
    os.path.join(MODEL_DIR, "vision_model_final.pt"),
    MODEL_DIR,
    DEVICE
)

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
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img)

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

        brand_logits = outputs["brand_output"]
        model_logits = outputs["model_output"]
        cond_logits = outputs["condition_output"]


        brand_pred = torch.softmax(brand_logits, dim=1).cpu().numpy()
        model_pred = torch.softmax(model_logits, dim=1).cpu().numpy()
        cond_pred = torch.softmax(cond_logits, dim=1).cpu().numpy()

        for j, listing_id in enumerate(ids):
            rows.append({
                "listing_id": listing_id,
                "pred_brand": brand_classes[np.argmax(brand_pred[j])],
                "pred_model": model_classes[np.argmax(model_pred[j])],
                "pred_condition": condition_classes[np.argmax(cond_pred[j])],
                "brand_conf": float(np.max(brand_pred[j])),
                "model_conf": float(np.max(model_pred[j])),
                "condition_conf": float(np.max(cond_pred[j]))
            })

        print(f"Processed {min(i + BATCH_SIZE, len(df))}/{len(df)}")

pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Saved predictions: {OUTPUT_FILE}")