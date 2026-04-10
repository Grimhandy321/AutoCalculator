import os
import sys
import joblib
import numpy as np

MODEL_DIR = "../models"

price_model = joblib.load(os.path.join(MODEL_DIR, "price_model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "price_encoders.pkl"))

def safe_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    return 0

if len(sys.argv) < 10:
    print("Usage:")
    print("python without_images.py year mileage fuel gearbox power_kw body_type brand model condition")
    sys.exit(1)

year = float(sys.argv[1])
mileage_km = float(sys.argv[2])
fuel = sys.argv[3]
gearbox = sys.argv[4]
power_kw = float(sys.argv[5])
body_type = sys.argv[6]
brand = sys.argv[7]
model = sys.argv[8]
condition = sys.argv[9]

X = [[
    year,
    mileage_km,
    power_kw,
    safe_transform(encoders["fuel"], fuel),
    safe_transform(encoders["gearbox"], gearbox),
    safe_transform(encoders["body_type"], body_type),
    safe_transform(encoders["pred_brand"], brand),
    safe_transform(encoders["pred_model"], model),
    safe_transform(encoders["pred_condition"], condition),
    1.0,  # fake
    1.0,
    1.0
]]

pred_log = price_model.predict(X)[0]
pred_price = np.expm1(pred_log)

print("Predicted price (no vision):", round(pred_price, 0), "CZK")