import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR = "../models"
DATA_FILE = "../data/bazos_cars_labeled.csv"
IMAGE_DIR = "../data/car_images"

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(DATA_FILE, low_memory=False)
df.columns = df.columns.str.strip()


required = ["listing_id", "brand", "model_extracted", "condition"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df = df.dropna(subset=required)

brand_classes = np.load(os.path.join(MODEL_DIR, "brand_classes.npy"), allow_pickle=True)
model_classes = np.load(os.path.join(MODEL_DIR, "model_classes.npy"), allow_pickle=True)
condition_classes = np.load(os.path.join(MODEL_DIR, "condition_classes.npy"), allow_pickle=True)


class VisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
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
        return (
            self.brand_output(x),
            self.model_output(x),
            self.condition_output(x),
        )


model = VisionModel().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "vision_model_final.pt"), map_location=DEVICE))
model.eval()


def find_image_path(listing_id):
    listing_id = str(listing_id)
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = os.path.join(IMAGE_DIR, f"{listing_id}{ext}")
        if os.path.exists(path):
            return path
    return None


df["image_path"] = df["listing_id"].apply(find_image_path)
df = df.dropna(subset=["image_path"]).reset_index(drop=True)

df = pd.concat([
    group.sample(n=min(len(group), 50), random_state=42)
    for _, group in df.groupby("brand", sort=False)
]).reset_index(drop=True)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_image(path):
    return transform(Image.open(path).convert("RGB"))


images = torch.stack([load_image(p) for p in df["image_path"]]).to(DEVICE)

with torch.no_grad():
    brand_logits, model_logits, cond_logits = model(images)

brand_pred = torch.argmax(brand_logits, 1).cpu().numpy()
model_pred = torch.argmax(model_logits, 1).cpu().numpy()
cond_pred = torch.argmax(cond_logits, 1).cpu().numpy()


def make_map(classes):
    return {v: i for i, v in enumerate(classes)}


brand_map = make_map(brand_classes)
model_map = make_map(model_classes)
cond_map = make_map(condition_classes)

brand_true = np.array([brand_map.get(x, -1) for x in df["brand"]])
model_true = np.array([model_map.get(x, -1) for x in df["model_extracted"]])
cond_true = np.array([cond_map.get(x, -1) for x in df["condition"]])

brand_mask = brand_true >= 0
model_mask = model_true >= 0
cond_mask = cond_true >= 0

print("=== BRAND REPORT ===")
print(classification_report(
    brand_true[brand_mask],
    brand_pred[brand_mask],
    labels=range(len(brand_classes)),
    target_names=brand_classes,
    zero_division=0
))

print("=== MODEL REPORT ===")
print(classification_report(
    model_true[model_mask],
    model_pred[model_mask],
    labels=range(len(model_classes)),
    target_names=model_classes,
    zero_division=0
))

print("=== CONDITION REPORT ===")
print(classification_report(
    cond_true[cond_mask],
    cond_pred[cond_mask],
    labels=range(len(condition_classes)),
    target_names=condition_classes,
    zero_division=0
))

print("=== CONFUSION MATRIX (CONDITION) ===")
cm = confusion_matrix(cond_true[cond_mask], cond_pred[cond_mask])
print(pd.DataFrame(cm, index=condition_classes, columns=condition_classes))


print("=== BRAND CONFUSION MATRIX ===")
cm = confusion_matrix(brand_true, brand_pred, labels=list(range(len(brand_classes))))
df_cm = pd.DataFrame(
    cm,
    index=brand_classes,
    columns=brand_classes
)

print(df_cm)