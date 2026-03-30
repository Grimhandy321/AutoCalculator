import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

CSV_FILE = "../data/bazos_cars_labeled.csv"
IMAGE_DIR = "../data/car_images"
MODEL_DIR = "../models"
OUTPUT_FILE = "../data/vision_predictions.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

df = pd.read_csv(CSV_FILE)
df["listing_id"] = df["listing_id"].astype(str)

brand_classes = np.load(os.path.join(MODEL_DIR, "brand_classes.npy"), allow_pickle=True)
model_classes = np.load(os.path.join(MODEL_DIR, "model_classes.npy"), allow_pickle=True)
condition_classes = np.load(os.path.join(MODEL_DIR, "condition_classes.npy"), allow_pickle=True)

model = load_model(os.path.join(MODEL_DIR, "vision_model_final.keras"))

def find_image_path(listing_id):
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = os.path.join(IMAGE_DIR, f"{listing_id}{ext}")
        if os.path.exists(path):
            return path
    return None

df["image_path"] = df["listing_id"].apply(find_image_path)
df = df.dropna(subset=["image_path"]).reset_index(drop=True)

def load_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

rows = []

for i in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[i:i+BATCH_SIZE]
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

    images = np.array(images)
    brand_pred, model_pred, condition_pred = model.predict(images, verbose=0)

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

    print(f"Processed {min(i+BATCH_SIZE, len(df))}/{len(df)}")

pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Saved predictions: {OUTPUT_FILE}")