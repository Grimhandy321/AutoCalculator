import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder

from xgboost import XGBRegressor

DATA_FILE = "../data/bazos_cars_labeled.csv"
VISION_FILE = "../data/vision_predictions.csv"
MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_FILE)
vision_df = pd.read_csv(VISION_FILE)

df["listing_id"] = df["listing_id"].astype(str)
vision_df["listing_id"] = vision_df["listing_id"].astype(str)

df = df.merge(vision_df, on="listing_id", how="inner")

print(df.shape)

# =========================
# CLEANING
# =========================
for col in ["price_czk", "year", "mileage_km", "power_kw"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["price_czk"])
df = df[(df["price_czk"] > 10000) & (df["price_czk"] < 5000000)]

# remove outliers
df = df[df["mileage_km"] < 400000]
df = df[df["power_kw"] < 500]

# fill numeric
for col in ["year", "mileage_km", "power_kw"]:
    df[col] = df[col].fillna(df[col].median())

# fill categorical
cat_cols = [
    "fuel", "transmission", "body_type",
    "pred_brand", "pred_model", "pred_condition"
]

for col in cat_cols:
    df[col] = df[col].fillna("unknown").astype(str)

# SAFE FEATURE ENGINEERING
df["car_age"] = 2025 - df["year"]

# Fix invalid ages
df["car_age"] = df["car_age"].clip(lower=0, upper=50)

# Safe division
df["km_per_year"] = df["mileage_km"] / (df["car_age"] + 1)

# Log transform safely
df["mileage_km"] = df["mileage_km"].clip(lower=0)
df["log_mileage"] = np.log1p(df["mileage_km"])


print(df.columns)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(df.shape)

# FEATURES
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
y = np.log1p(df["price_czk"])



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

joblib.dump((X_test.index, y_test.index), os.path.join(MODEL_DIR, "test_idx.pkl"))

categorical_features = [
    "fuel", "transmission",
    "pred_brand", "pred_model", "pred_condition"
]

preprocessor = ColumnTransformer([
    ("cat", TargetEncoder(), categorical_features)
], remainder="passthrough")

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=10,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# =========================
# TRAIN
# =========================
pipeline.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
pred_log = pipeline.predict(X_test)
pred = np.expm1(pred_log)
actual = np.expm1(y_test)

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2 = r2_score(actual, pred)
mape = mean_absolute_percentage_error(actual, pred)

print("\n=== TRAINING RESULTS ===")
print(f"MAE  : {mae:.2f} CZK")
print(f"RMSE : {rmse:.2f} CZK")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape * 100:.2f} %")

# =========================
# CROSS VALIDATION
# =========================
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print(f"CV MAE: {-scores.mean():.2f} CZK")

# =========================
# SAVE
# =========================
joblib.dump(pipeline, os.path.join(MODEL_DIR, "price_pipeline.pkl"))
print("Saved pipeline.")