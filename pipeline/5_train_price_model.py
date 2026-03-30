import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

DATA_FILE = "../data/bazos_cars_labeled.csv"
VISION_FILE = "../data/vision_predictions.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)
vision_df = pd.read_csv(VISION_FILE)

df["listing_id"] = df["listing_id"].astype(str)
vision_df["listing_id"] = vision_df["listing_id"].astype(str)

df = df.merge(vision_df, on="listing_id", how="inner")

# -----------------------------
# Clean data
# -----------------------------
for col in ["price_czk", "year", "mileage_km", "power_kw"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["price_czk"])
df = df[df["price_czk"] > 10000]
df = df[df["price_czk"] < 5000000]

df["year"] = df["year"].fillna(df["year"].median())
df["mileage_km"] = df["mileage_km"].fillna(df["mileage_km"].median())
df["power_kw"] = df["power_kw"].fillna(df["power_kw"].median())

for col in ["fuel", "gearbox", "body_type", "pred_brand", "pred_model", "pred_condition"]:
    df[col] = df[col].fillna("unknown").astype(str)

# -----------------------------
# Encode categoricals
# -----------------------------
encoders = {}
cat_cols = ["fuel", "gearbox", "body_type", "pred_brand", "pred_model", "pred_condition"]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

joblib.dump(encoders, os.path.join(MODEL_DIR, "price_encoders.pkl"))

# -----------------------------
# Features / target
# -----------------------------
feature_cols = [
    "year", "mileage_km", "power_kw",
    "fuel", "gearbox", "body_type",
    "pred_brand", "pred_model", "pred_condition",
    "brand_conf", "model_conf", "condition_conf"
]

X = df[feature_cols]
y = np.log1p(df["price_czk"])  # log transform improves regression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train XGBoost (parallelized)
# -----------------------------
model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    n_jobs=-1,   # <-- parallel CPU usage
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
pred_log = model.predict(X_test)
pred = np.expm1(pred_log)
actual = np.expm1(y_test)

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2 = r2_score(actual, pred)

print("\n=== PRICE MODEL RESULTS ===")
print(f"MAE  : {mae:.2f} CZK")
print(f"RMSE : {rmse:.2f} CZK")
print(f"R²   : {r2:.4f}")

joblib.dump(model, os.path.join(MODEL_DIR, "price_model.pkl"))
print("Saved price model.")