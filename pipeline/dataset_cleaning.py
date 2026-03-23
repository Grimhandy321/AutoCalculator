import json
import pandas as pd
import numpy as np

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_data(data):
    df = pd.DataFrame(data)
    df = df.dropna(subset=["price", "km", "year"])

    df = df[(df["price"] > 20000) & (df["price"] < 5000000)]
    df = df[(df["km"] > 0) & (df["km"] < 500000)]
    df = df[(df["year"] > 1995) & (df["year"] <= 2025)]

    df["car_age"] = 2025 - df["year"]
    df["km_per_year"] = df["km"] / (df["car_age"] + 1)

    df["price_log"] = np.log1p(df["price"])
    df["km_log"] = np.log1p(df["km"])

    df["fuel"] = df["fuel"].fillna("Unknown")
    df["transmission"] = df["transmission"].fillna("Unknown")
    df["engine_type"] = df["engine_type"].fillna("Unknown")

    return df

def save_clean(df):
    df.to_csv("data/processed/cars_clean.csv", index=False)
