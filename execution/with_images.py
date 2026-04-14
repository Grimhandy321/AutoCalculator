import os
import sys
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

def prepare_image(path):
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)

def safe_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    return 0

if len(sys.argv) < 8:
    print("Usage:")
    print("python with_images.py image.jpg year mileage fuel gearbox power_kw body_type")
    sys.exit(1)

image_path = sys.argv[1]
year = float(sys.argv[2])
mileage_km = float(sys.argv[3])
fuel = sys.argv[4]
gearbox = sys.argv[5]
power_kw = float(sys.argv[6])
body_type = sys.argv[7]

img = prepare_image(image_path)

with torch.no_grad():
    brand_pred, model_pred, condition_pred = vision_model(img)

brand_pred = torch.softmax(brand_pred, 1).cpu().numpy()[0]
model_pred = torch.softmax(model_pred, 1).cpu().numpy()[0]
condition_pred = torch.softmax(condition_pred, 1).cpu().numpy()[0]

pred_brand = brand_classes[np.argmax(brand_pred)]
pred_model = model_classes[np.argmax(model_pred)]
pred_condition = condition_classes[np.argmax(condition_pred)]

X = [[
    year,
    mileage_km,
    power_kw,
    safe_transform(encoders["fuel"], fuel),
    safe_transform(encoders["gearbox"], gearbox),
    safe_transform(encoders["body_type"], body_type),
    safe_transform(encoders["pred_brand"], pred_brand),
    safe_transform(encoders["pred_model"], pred_model),
    safe_transform(encoders["pred_condition"], pred_condition),
    float(np.max(brand_pred)),
    float(np.max(model_pred)),
    float(np.max(condition_pred))
]]

pred_log = price_model.predict(X)[0]
pred_price = np.expm1(pred_log)

print("Predicted price:", round(pred_price, 0), "CZK")