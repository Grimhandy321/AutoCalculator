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

# Load model
model = load_model(os.path.join(MODEL_DIR, "vision_model_final.keras"))

# Load class mappings
brand_classes = np.load(os.path.join(MODEL_DIR, "brand_classes.npy"), allow_pickle=True)
model_classes = np.load(os.path.join(MODEL_DIR, "model_classes.npy"), allow_pickle=True)
condition_classes = np.load(os.path.join(MODEL_DIR, "condition_classes.npy"), allow_pickle=True)

# Add UNKNOWN
brand_classes_ext = np.append(brand_classes, "UNKNOWN")
model_classes_ext = np.append(model_classes, "UNKNOWN")
condition_classes_ext = np.append(condition_classes, "UNKNOWN")

# Load data
df = pd.read_csv(DATA_FILE)


def find_image_path(listing_id):
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = os.path.join(IMAGE_DIR, f"{listing_id}{ext}")
        if os.path.exists(path):
            return path
    return None


df["image_path"] = df["listing_id"].astype(str).apply(find_image_path)
df = df.dropna(subset=["image_path"]).head(500)


# Image loader
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


def encode_with_classes(values, classes):
    mapping = {v: i for i, v in enumerate(classes)}
    return np.array([mapping.get(v, len(classes)) for v in values])  # UNKNOWN = last index


brand_true = encode_with_classes(df["brand"], brand_classes)
model_true = encode_with_classes(df["model_extracted"], model_classes)
condition_true = encode_with_classes(df["condition"], condition_classes)


def safe_pred(pred, classes):
    return np.array([p if p < len(classes) else len(classes) for p in pred])


brand_pred = safe_pred(brand_pred, brand_classes)
model_pred = safe_pred(model_pred, model_classes)
condition_pred = safe_pred(condition_pred, condition_classes)

# Reports
print("=== BRAND REPORT ===")
labels_brand = list(range(len(brand_classes_ext)))

print(classification_report(
    brand_true, brand_pred,
    labels=labels_brand,
    target_names=brand_classes_ext,
    zero_division=0
))

print("=== MODEL REPORT ===")
labels_model = list(range(len(model_classes_ext)))

print(classification_report(
    model_true, model_pred,
    labels=labels_model,
    target_names=model_classes_ext,
    zero_division=0
))

print("=== CONDITION REPORT ===")
labels_condition = list(range(len(condition_classes_ext)))

print(classification_report(
    condition_true, condition_pred,
    labels=labels_condition,
    target_names=condition_classes_ext,
    zero_division=0
))

print("=== CONFUSION MATRIX (CONDITION) ===")
labels_condition = list(range(len(condition_classes_ext)))

cm = confusion_matrix(
    condition_true,
    condition_pred,
    labels=labels_condition
)
df_cm = pd.DataFrame(cm, index=condition_classes_ext, columns=condition_classes_ext)
print(df_cm)

print("\n=== SAMPLE PREDICTIONS ===")
for i in range(min(10, len(df))):
    print("----")
    print("TRUE:",
          brand_classes_ext[brand_true[i]],
          model_classes_ext[model_true[i]],
          condition_classes_ext[condition_true[i]])

    print("PRED:",
          brand_classes_ext[brand_pred[i]],
          model_classes_ext[model_pred[i]],
          condition_classes_ext[condition_pred[i]])