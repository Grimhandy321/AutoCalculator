import os
import json
import io
import torch
import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

from pipeline.models.vision_model import VisionModel

CURRENT_YEAR = 2026
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/frontend")
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

brand_classes = np.load("models/brand_classes.npy", allow_pickle=True)
model_classes = np.load("models/model_classes.npy", allow_pickle=True)
cond_classes = np.load("models/condition_classes.npy", allow_pickle=True)
brand_model_mask = np.load("models/brand_model_mask.npy")

vision_model = VisionModel(
    len(brand_classes),
    len(model_classes),
    len(cond_classes)
).to(DEVICE)

vision_model.load_state_dict(torch.load("models/vision_model_final.pt", map_location=DEVICE))
vision_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

price_model = joblib.load("models/price_pipeline.pkl")

@app.route("/")
def serve_index():
    return app.send_static_file("index.html")

@app.route("/data")
def get_data():
    return jsonify({"brands": brand_classes.tolist()})

@app.route("/models/<brand_name>")
def get_models(brand_name):
    if brand_name not in brand_classes:
        return jsonify({"models": []})

    brand_idx = np.where(brand_classes == brand_name)[0][0]
    mask = brand_model_mask[brand_idx]

    valid_models = [
        model_classes[i]
        for i in range(len(model_classes))
        if mask[i] == 1
    ]

    return jsonify({"models": valid_models})

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = vision_model(image)

        brand_idx = outputs["brand_output"].argmax(1).item()

        mask = brand_model_mask[brand_idx]
        model_logits = outputs["model_output"].cpu().numpy()[0]
        model_logits[mask == 0] = -1e9
        model_idx = model_logits.argmax()

    brand = brand_classes[brand_idx]
    model_name = model_classes[model_idx]
    brand_probs = torch.softmax(outputs["brand_output"], dim=1)
    confidence = brand_probs.max().item()

    return jsonify({
        "brand": brand,
        "model": model_name,
        "confidence": confidence
    })

@app.route("/price", methods=["POST"])
def get_price():
    data = request.get_json()

    mileage = float(data.get("mileage", 0))
    year = int(data.get("year", CURRENT_YEAR))
    fuel = str(data.get("engine", "unknown"))
    transmission = "manual"
    brand = str(data.get("brand"))
    model_name = str(data.get("model"))

    car_age = max(0, CURRENT_YEAR - year)
    log_mileage = np.log1p(max(mileage, 0))

    X = pd.DataFrame([{
        "car_age": car_age,
        "mileage_km": mileage,
        "log_mileage": log_mileage,
        "fuel": fuel,
        "transmission": transmission,
        "pred_brand": brand,
        "pred_model": model_name
    }]).astype({
        "fuel": "object",
        "transmission": "object",
        "pred_brand": "object",
        "pred_model": "object"
    })

    pred_log = price_model.predict(X)[0]
    price = int(np.expm1(pred_log))

    return jsonify({
        "price": price,
        "car_age": car_age  # optional debug info
    })


@app.route("/evaluation")
def get_evaluation():
    with open("data/vision_eval_report.json", "r") as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
