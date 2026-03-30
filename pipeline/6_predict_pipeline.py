import os
import sys
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_DIR = "../models"
IMG_SIZE = (224, 224)

vision_model = load_model(os.path.join(MODEL_DIR, "vision_model_final.keras"))
price_model = joblib.load(os.path.join(MODEL_DIR, "price_model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "price_encoders.pkl"))

brand_classes = np.load(os.path.join(MODEL_DIR, "brand_classes.npy"), allow_pickle=True)
model_classes = np.load(os.path.join(MODEL_DIR, "model_classes.npy"), allow_pickle=True)
condition_classes = np.load(os.path.join(MODEL_DIR, "condition_classes.npy"), allow_pickle=True)

def prepare_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

def safe_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    return 0

if len(sys.argv) < 2:
    print("Usage:")
    print("python 6_predict_pipeline.py image.jpg year mileage fuel gearbox power_kw body_type")
    sys.exit(1)

image_path = sys.argv[1]
year = float(sys.argv[2])
mileage_km = float(sys.argv[3])
fuel = sys.argv[4]
gearbox = sys.argv[5]
power_kw = float(sys.argv[6])
body_type = sys.argv[7]

img = prepare_image(image_path)
brand_pred, model_pred, condition_pred = vision_model.predict(img, verbose=0)

pred_brand = brand_classes[np.argmax(brand_pred[0])]
pred_model = model_classes[np.argmax(model_pred[0])]
pred_condition = condition_classes[np.argmax(condition_pred[0])]

brand_conf = float(np.max(brand_pred[0]))
model_conf = float(np.max(model_pred[0]))
condition_conf = float(np.max(condition_pred[0]))

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
    brand_conf,
    model_conf,
    condition_conf
]]

pred_log = price_model.predict(X)[0]
pred_price = np.expm1(pred_log)

print("\n=== VISION OUTPUT ===")
print(f"Predicted brand     : {pred_brand} ({brand_conf:.2f})")
print(f"Predicted model     : {pred_model} ({model_conf:.2f})")
print(f"Predicted condition : {pred_condition} ({condition_conf:.2f})")

print("\n=== PRICE PREDICTION ===")
print(f"Predicted price: {pred_price:,.0f} CZK")