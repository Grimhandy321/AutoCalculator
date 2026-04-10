import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

MODEL_DIR = "../models"
DATA_FILE = "../data/bazos_cars_labeled.csv"
VISION_FILE = "../data/vision_predictions.csv"

model = joblib.load(os.path.join(MODEL_DIR, "price_model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "price_encoders.pkl"))

df = pd.read_csv(DATA_FILE)
vision_df = pd.read_csv(VISION_FILE)

df["listing_id"] = df["listing_id"].astype(str)
vision_df["listing_id"] = vision_df["listing_id"].astype(str)

df = df.merge(vision_df, on="listing_id")

# Clean data
for col in ["price_czk", "year", "mileage_km", "power_kw"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["price_czk"])
df = df[df["price_czk"] > 10000]
df = df[df["price_czk"] < 5000000]

df["year"] = df["year"].fillna(df["year"].median())
df["mileage_km"] = df["mileage_km"].fillna(df["mileage_km"].median())
df["power_kw"] = df["power_kw"].fillna(df["power_kw"].median())

for col in ["fuel", "gearbox", "pred_brand", "pred_model", "pred_condition", "body_type"]:
    df[col] = df[col].fillna("unknown").astype(str)

# Encoding
for col, le in encoders.items():
    df[col] = df[col].map(lambda x: x if x in le.classes_ else "unknown")
    df[col] = le.transform(df[col])

# Features
feature_cols = [
    "year", "mileage_km", "power_kw",
    "fuel", "gearbox", "body_type",
    "pred_brand", "pred_model", "pred_condition",
    "brand_conf", "model_conf", "condition_conf"
]

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["price_czk"]

# Predict
pred_log = model.predict(X)
pred = np.expm1(pred_log)

# Metrics
mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))
r2 = r2_score(y, pred)
mape = mean_absolute_percentage_error(y, pred)

print("=== PRICE MODEL METRICS ===")
print(f"MAE  : {mae:.2f} CZK")
print(f"RMSE : {rmse:.2f} CZK")
print(f"R2   : {r2:.4f}")
print(f"MAPE : {mape * 100:.2f} %")

# Plots
plt.scatter(y, pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

plt.hist(y - pred, bins=50)
plt.title("Residual Distribution")
plt.show()