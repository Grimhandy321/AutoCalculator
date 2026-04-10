
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_DIR = "../models"
DATA_FILE = "../data/bazos_cars_labeled.csv"
IMAGE_DIR = "../data/car_images"
IMG_SIZE = (224, 224)

model = load_model(os.path.join(MODEL_DIR, "vision_model_final.keras"))

brand_classes = np.load(os.path.join(MODEL_DIR, "brand_classes.npy"), allow_pickle=True)
model_classes = np.load(os.path.join(MODEL_DIR, "model_classes.npy"), allow_pickle=True)
condition_classes = np.load(os.path.join(MODEL_DIR, "condition_classes.npy"), allow_pickle=True)

df = pd.read_csv(DATA_FILE)

def find_image_path(listing_id):
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = os.path.join(IMAGE_DIR, f"{listing_id}{ext}")
        if os.path.exists(path):
            return path
    return None

df["image_path"] = df["listing_id"].astype(str).apply(find_image_path)
df = df.dropna(subset=["image_path"]).head(500)

def load_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

images = np.array([load_image(p) for p in df["image_path"]])

preds = model.predict(images, verbose=1)
brand_pred = np.argmax(preds[0], axis=1)
model_pred = np.argmax(preds[1], axis=1)
condition_pred = np.argmax(preds[2], axis=1)

# Encode true labels
brand_true = pd.factorize(df["brand"])[0]
model_true = pd.factorize(df["model_extracted"])[0]
condition_true = pd.factorize(df["condition"])[0]

print("=== BRAND REPORT ===")
print(classification_report(brand_true, brand_pred))

print("=== MODEL REPORT ===")
print(classification_report(model_true, model_pred))

print("=== CONDITION REPORT ===")
print(classification_report(condition_true, condition_pred))

print("=== CONFUSION MATRIX (CONDITION) ===")
print(confusion_matrix(condition_true, condition_pred))