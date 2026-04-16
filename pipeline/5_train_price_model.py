import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from xgboost import XGBRegressor

DATA_FILE = "../data/bazos_cars_labeled.csv"
VISION_FILE = "../data/vision_predictions.csv"
MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)
vision_df = pd.read_csv(VISION_FILE)

df["listing_id"] = df["listing_id"].astype(str)
vision_df["listing_id"] = vision_df["listing_id"].astype(str)

df = df.merge(vision_df, on="listing_id", how="inner")

df["price_czk"] = pd.to_numeric(df["price_czk"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")

df = df.dropna(subset=["price_czk"])
df = df[(df["price_czk"] > 10000) & (df["price_czk"] < 5000000)]

df["year"] = df["year"].fillna(df["year"].median())
df["mileage_km"] = df["mileage_km"].fillna(df["mileage_km"].median())

for col in ["fuel", "transmission", "pred_brand", "pred_model"]:
    df[col] = df[col].fillna("unknown").astype(str)

df["mileage_km"] = df["mileage_km"].clip(lower=0)
df["log_mileage"] = np.log1p(df["mileage_km"])

feature_cols = [
    "year",
    "mileage_km",
    "log_mileage",
    "fuel",
    "transmission",
    "pred_brand",
    "pred_model"
]

X = df[feature_cols]
y = np.log1p(df["price_czk"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_features = [
    "fuel", "transmission",
    "pred_brand", "pred_model"
]

preprocessor = ColumnTransformer([
    ("cat", TargetEncoder(), categorical_features)
], remainder="passthrough")

model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, os.path.join(MODEL_DIR, "price_pipeline.pkl"))
