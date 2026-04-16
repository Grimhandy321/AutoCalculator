import os
import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models

MODEL_DIR = "../models"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

price_model = joblib.load(os.path.join(MODEL_DIR, "price_model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "price_encoders.pkl"))

brand_classes = np.load(os.path.join(MODEL_DIR, "brand_classes.npy"), allow_pickle=True)
model_classes = np.load(os.path.join(MODEL_DIR, "model_classes.npy"), allow_pickle=True)
condition_classes = np.load(os.path.join(MODEL_DIR, "condition_classes.npy"), allow_pickle=True)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

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

vision_model = VisionModel().to(DEVICE)
vision_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "vision_model_final.pt"), map_location=DEVICE))
vision_model.eval()

def safe_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    return 0

def predict_from_image(path):
    img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        b, m, c = vision_model(img)

    b = torch.softmax(b, 1).cpu().numpy()[0]
    m = torch.softmax(m, 1).cpu().numpy()[0]
    c = torch.softmax(c, 1).cpu().numpy()[0]

    return (
        brand_classes[np.argmax(b)],
        model_classes[np.argmax(m)],
        condition_classes[np.argmax(c)],
        float(np.max(b)),
        float(np.max(m)),
        float(np.max(c))
    )

print("=== STEP 1: IMAGE ===")
image_path = input("Enter image path: ").strip()

pred_brand, pred_model, pred_condition, bc, mc, cc = predict_from_image(image_path)

print("\n=== MODEL SUGGESTION ===")
print(f"Brand: {pred_brand} ({bc:.2f})")
print(f"Model: {pred_model} ({mc:.2f})")
print(f"Condition: {pred_condition} ({cc:.2f})")

print("\n=== STEP 2: CONFIRM / EDIT ===")
brand = input(f"Brand [{pred_brand}]: ") or pred_brand
model = input(f"Model [{pred_model}]: ") or pred_model
condition = input(f"Condition [{pred_condition}]: ") or pred_condition

print("\n=== STEP 3: OTHER INFO ===")
year = float(input("Year: "))
mileage_km = float(input("Mileage (km): "))
fuel = input("Fuel: ")
gearbox = input("Gearbox: ")
power_kw = float(input("Power (kW): "))

X = [[
    year,
    mileage_km,
    power_kw,
    safe_transform(encoders["fuel"], fuel),
    safe_transform(encoders["gearbox"], gearbox),
    safe_transform(encoders["pred_brand"], brand),
    safe_transform(encoders["pred_model"], model),
    safe_transform(encoders["pred_condition"], condition),
    bc,
    mc,
    cc
]]

pred_log = price_model.predict(X)[0]
pred_price = np.expm1(pred_log)

print("\n=== RESULT ===")
print("Predicted price:", round(pred_price, 0), "CZK")