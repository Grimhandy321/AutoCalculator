import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from data.data_cleaner import full_cleaning_pipeline

# 1. Load Data
DATA_PATH = "../data/autojarov_cars.csv"
MODEL_OUTPUT_PATH = "model.pkl"

df = pd.read_csv(DATA_PATH)



print("Loaded dataset shape:", df.shape)
df = full_cleaning_pipeline(df)
print("After cleaning:", df.shape)


# 3. Feature / Target Split
X = df.drop("price", axis=1)
y = df["price"]


# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# 5. Preprocessing
categorical_features = [
    "brand",
    # "engine_type",
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


# 6. Model Definition (XGBoost)
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    objective="reg:squarederror"
)


# 7. Full Pipeline
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", model)
])


# 8. Train
pipeline.fit(X_train, y_train)


# 9. Evaluate
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n=== Evaluation ===")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:,.0f} Kč")


# 10. Save Model
with open(MODEL_OUTPUT_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print("\nSaved to:", MODEL_OUTPUT_PATH)
