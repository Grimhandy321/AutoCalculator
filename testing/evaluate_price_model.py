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

# LOAD PIPELINE
pipeline = joblib.load(os.path.join(MODEL_DIR, "price_pipeline.pkl"))
test_idx, _ = joblib.load(os.path.join(MODEL_DIR, "test_idx.pkl"))
# LOAD DATA
df = pd.read_csv(DATA_FILE)
vision_df = pd.read_csv(VISION_FILE)

df["listing_id"] = df["listing_id"].astype(str)
vision_df["listing_id"] = vision_df["listing_id"].astype(str)

df = df.merge(vision_df, on="listing_id", how="inner")

for col in ["price_czk", "year", "mileage_km", "power_kw"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["price_czk"])
df = df[(df["price_czk"] > 10000) & (df["price_czk"] < 5000000)]

df = df[df["mileage_km"] < 400000]
df = df[df["power_kw"] < 500]

for col in ["year", "mileage_km", "power_kw"]:
    df[col] = df[col].fillna(df[col].median())

cat_cols = [
    "fuel", "transmission", "body_type",
    "pred_brand", "pred_model", "pred_condition"
]

for col in cat_cols:
    df[col] = df[col].fillna("unknown").astype(str)

# SAME FEATURES
df["car_age"] = 2025 - df["year"]
df["km_per_year"] = df["mileage_km"] / (df["car_age"] + 1)
df["log_mileage"] = np.log1p(df["mileage_km"])

feature_cols = [
    "year",
    "mileage_km",
    "power_kw",
    "car_age",
    "km_per_year",
    "log_mileage",
    "fuel",
    "transmission",
    "pred_brand",
    "pred_model",
    "pred_condition",
    "brand_conf",
    "model_conf",
    "condition_conf"
]

X = df[feature_cols]
y = df["price_czk"]


X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

# PREDICT
pred_log = pipeline.predict(X_test)
pred = np.expm1(pred_log)

# METRICS
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
mape = mean_absolute_percentage_error(y_test, pred)

print("\n=== TEST RESULTS ===")
print(f"MAE  : {mae:.2f} CZK")
print(f"RMSE : {rmse:.2f} CZK")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape * 100:.2f} %")


# PLOTS
plt.scatter(y_test, pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()

perc_error = (y_test - pred) / y_test
log_perc_error = np.log1p(np.abs(perc_error))

plt.hist(log_perc_error, bins=50)
plt.title("Log Percentage Error Distribution")
plt.xlabel("log1p(|percentage error|)")
plt.show()