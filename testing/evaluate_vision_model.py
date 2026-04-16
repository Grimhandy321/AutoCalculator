import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score

from pipeline.models.vision_model import load_vision_model

CSV_FILE = "../data/bazos_cars_labeled.csv"
IMAGE_DIR = "../data/car_images"
MODEL_PATH = "../models/vision_model_final.pt"
MODEL_DIR = "../models"
OUTPUT_JSON = "../data/vision_eval_report.json"

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Starting evaluation...")
print(f"Device: {DEVICE}")


model = load_vision_model(
    MODEL_PATH,
    len(np.load(os.path.join(MODEL_DIR, "brand_classes.npy"), allow_pickle=True)),
    len(np.load(os.path.join(MODEL_DIR, "model_classes.npy"), allow_pickle=True)),
    len(np.load(os.path.join(MODEL_DIR, "condition_classes.npy"), allow_pickle=True)),
    DEVICE
)

print("Model loaded")

brand_classes = np.load(os.path.join(MODEL_DIR, "brand_classes.npy"), allow_pickle=True)
model_classes = np.load(os.path.join(MODEL_DIR, "model_classes.npy"), allow_pickle=True)
condition_classes = np.load(os.path.join(MODEL_DIR, "condition_classes.npy"), allow_pickle=True)


print("Loading dataset...")

df = pd.read_csv(CSV_FILE, low_memory=False)
df["listing_id"] = df["listing_id"].astype(str)

print("Rows loaded:", len(df))

def find_image_path(listing_id):
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = os.path.join(IMAGE_DIR, f"{listing_id}{ext}")
        if os.path.exists(path):
            return path
    return None

df["image_path"] = df["listing_id"].apply(find_image_path)
df = df.dropna(subset=["image_path"]).reset_index(drop=True)

print("Valid images:", len(df))

# take small sample for evaluation speed
df = df.sample(min(500, len(df)), random_state=42).reset_index(drop=True)

print("Evaluation sample size:", len(df))


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def load_image(path):
    return transform(Image.open(path).convert("RGB"))

print("Running inference...")

rows = []

all_brand_true = []
all_brand_pred = []

all_model_true = []
all_model_pred = []

all_cond_true = []
all_cond_pred = []

with torch.no_grad():
    for i in range(len(df)):
        if i % 50 == 0:
            print(f"Processing {i}/{len(df)}")

        row = df.iloc[i]

        try:
            img = load_image(row["image_path"]).unsqueeze(0).to(DEVICE)
        except Exception as e:
            print("Failed image:", row["image_path"])
            continue

        outputs = model(img)

        brand_logits = outputs["brand_output"]
        model_logits = outputs["model_output"]
        cond_logits = outputs["condition_output"]

        brand_pred = torch.argmax(brand_logits, dim=1).item()
        model_pred = torch.argmax(model_logits, dim=1).item()
        cond_pred = torch.argmax(cond_logits, dim=1).item()

        # true labels
        brand_true = np.where(brand_classes == row["brand"])[0]
        model_true = np.where(model_classes == row["model_extracted"])[0]
        cond_true = np.where(condition_classes == row["condition"])[0]

        if len(brand_true) == 0 or len(model_true) == 0 or len(cond_true) == 0:
            continue

        brand_true = brand_true[0]
        model_true = model_true[0]
        cond_true = cond_true[0]

        all_brand_true.append(brand_true)
        all_brand_pred.append(brand_pred)

        all_model_true.append(model_true)
        all_model_pred.append(model_pred)

        all_cond_true.append(cond_true)
        all_cond_pred.append(cond_pred)

        # save sample predictions
        if len(rows) < 10:
            rows.append({
                "listing_id": row["listing_id"],
                "true_brand": str(row["brand"]),
                "pred_brand": str(brand_classes[brand_pred]),
                "true_model": str(row["model_extracted"]),
                "pred_model": str(model_classes[model_pred]),
                "true_condition": str(row["condition"]),
                "pred_condition": str(condition_classes[cond_pred]),
            })

print("Computing metrics...")

brand_acc = accuracy_score(all_brand_true, all_brand_pred)
model_acc = accuracy_score(all_model_true, all_model_pred)
cond_acc = accuracy_score(all_cond_true, all_cond_pred)

print("Brand accuracy:", brand_acc)
print("Model accuracy:", model_acc)
print("Condition accuracy:", cond_acc)

brand_cm = confusion_matrix(all_brand_true, all_brand_pred)
model_cm = confusion_matrix(all_model_true, all_model_pred)
cond_cm = confusion_matrix(all_cond_true, all_cond_pred)


print("Saving ...")

def build_labeled_cm(y_true, y_pred, classes, name):
    cm = confusion_matrix(y_true, y_pred)

    return {
        "name": name,
        "labels": classes.tolist(),
        "matrix": cm.tolist()
    }

report = {
    "accuracy": {
        "brand": float(brand_acc),
        "model": float(model_acc),
        "condition": float(cond_acc),
    },

    "confusion_matrices": {
        "brand": build_labeled_cm(all_brand_true, all_brand_pred, brand_classes, "brand"),
        "model": build_labeled_cm(all_model_true, all_model_pred, model_classes, "model"),
        "condition": build_labeled_cm(all_cond_true, all_cond_pred, condition_classes, "condition"),
    },

    "samples": rows,
    "num_samples": len(all_brand_true)
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("Saved:", OUTPUT_JSON)
