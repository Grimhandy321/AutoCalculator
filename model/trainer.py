import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# 1. Load Data
DATA_PATH = "../data/autojarov_cars.csv"
MODEL_OUTPUT_PATH = "model.pkl"

df = pd.read_csv(DATA_PATH)

print("Loaded dataset shape:", df.shape)


# 2. Basic Cleaning
df = df.drop_duplicates()

df = df[df["km"] >= 0]
df = df[df["price"] > 0]
df = df[df["year"] > 1990]

df = df.dropna()

print("After cleaning:", df.shape)


# 3. Feature / Target Split
X = df.drop("price", axis=1)
y = df["price"]


# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# 5. Preprocessing
categorical_features = [
    "brand",
    "engine_type",
    "fuel",
    "transmission"
]

numeric_features = [
    "km",
    "year"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)


# 6. Model Definition
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)


# 7. Full Pipeline
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", model)
])



pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n=== Evaluation ===")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:,.0f} Kč")


# 10. Save Model
with open(MODEL_OUTPUT_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print("\nsaved to:", MODEL_OUTPUT_PATH)
