import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.models.vision_model import VisionModel

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD MODEL + DATA
brand_classes = np.load("models/brand_classes.npy", allow_pickle=True)
model_classes = np.load("models/model_classes.npy", allow_pickle=True)
cond_classes = np.load("models/condition_classes.npy", allow_pickle=True)
brand_model_mask = np.load("models/brand_model_mask.npy")

model = VisionModel(
    len(brand_classes),
    len(model_classes),
    len(cond_classes)
).to(DEVICE)

model.load_state_dict(torch.load("models/vision_model_final.pt", map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# PRICE LOGIC
def estimate_price(brand, model, condition):
    base_prices = {
        "BMW": 15000,
        "Audi": 14000,
        "Skoda": 8000,
        "Ford": 7000
    }

    condition_factor = {
        "new": 1.5,
        "used": 1.0,
        "damaged": 0.5
    }

    base = base_prices.get(brand, 6000)
    factor = condition_factor.get(condition.lower(), 1.0)

    return int(base * factor)

# =========================
# API ENDPOINTS
# =========================

@app.get("/data")
def get_data():
    return {
        "brands": brand_classes.tolist()
    }

@app.get("/models/{brand_name}")
def get_models_for_brand(brand_name: str):
    if brand_name not in brand_classes:
        return {"models": []}

    brand_idx = np.where(brand_classes == brand_name)[0][0]
    mask = brand_model_mask[brand_idx]

    valid_models = [
        model_classes[i]
        for i in range(len(model_classes))
        if mask[i] == 1
    ]

    return {"models": valid_models}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)

        brand_idx = outputs["brand_output"].argmax(1).item()

        # apply mask
        mask = brand_model_mask[brand_idx]
        model_logits = outputs["model_output"].cpu().numpy()[0]
        model_logits[mask == 0] = -1e9
        model_idx = model_logits.argmax()

        cond_idx = outputs["condition_output"].argmax(1).item()

    brand = brand_classes[brand_idx]
    model_name = model_classes[model_idx]
    condition = cond_classes[cond_idx]

    price = estimate_price(brand, model_name, condition)

    return {
        "brand": brand,
        "model": model_name,
        "price": price
    }

class PriceRequest(BaseModel):
    brand: str
    model: str

@app.post("/price")
def get_price(data: PriceRequest):
    condition = "used"  # hidden
    price = estimate_price(data.brand, data.model, condition)
    return {"price": price}