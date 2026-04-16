import re
import pandas as pd
from pipeline.utils.car_title_parser import clean_car_columns

CSV_FILE = "../data/bazos_cars_10k.csv"
LABELED_FILE = "../data/bazos_cars_labeled.csv"
CLEANED_FILE = "../data/data_cleaned.csv"

BAD_IMAGE_URL = "https://www.jasminka.cz/images/v/lecenizv.jpg"

df = pd.read_csv(CSV_FILE, low_memory=False)

# SAFE INIT
for col in ["title", "description", "brand", "model"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str)

df["price_czk"] = pd.to_numeric(df.get("price_czk"), errors="coerce")
df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
df["mileage_km"] = pd.to_numeric(df.get("mileage_km"), errors="coerce")
df["power_kw"] = pd.to_numeric(df.get("power_kw"), errors="coerce")

# CLEAN TEXT
def clean_text(s):
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

df["brand"] = df["brand"].apply(clean_text)
df["model"] = df["model"].apply(clean_text)


# FIX LISTING ID
df["listing_id"] = pd.to_numeric(df["listing_id"], errors="coerce")
df = df.dropna(subset=["listing_id"])
df = df[df["listing_id"] > 0]
df["listing_id"] = df["listing_id"].astype(int).astype(str)

# PRICE FILTER
df = df[df["price_czk"].between(10000, 2000000, inclusive="both")]

# REMOVE INVALID BRANDS
df = df[df["brand"].str.match(r"^[A-Za-zŠšČčŘřŽžÁáÉéÍíÓóÚúŮů]+", na=False)]

# CONDITION
def infer_condition(text):
    t = text.lower()

    poor_kw = ["na opravu", "nepojízdné", "bourané", "havárie", "poškozené", "na díly"]
    fair_kw = ["kosmetické vady", "horší stav", "menší opravy", "opotřebení"]
    excellent_kw = ["top stav", "perfektní", "jako nové", "bez investic", "garážované"]
    good_kw = ["zachovalé", "dobrý stav", "udržované", "pravidelný servis"]

    for kw in poor_kw:
        if kw in t:
            return "poor"
    for kw in fair_kw:
        if kw in t:
            return "fair"
    for kw in excellent_kw:
        if kw in t:
            return "excellent"
    for kw in good_kw:
        return "good"

    return "fair"

df["condition"] = (df["title"] + " " + df["description"]).apply(infer_condition)
df["condition"] = df["condition"].fillna("fair")

# IMAGE CLEAN
df["image_urls"] = df.get("image_urls", "").astype(str)
df["image_urls"] = df["image_urls"].str.replace(BAD_IMAGE_URL, "", regex=False)

# CLEAN STRUCTURE
df = clean_car_columns(df, title_col="title", model_col="model")

# SAFE COLUMN EXPORT
required_cols = [
    "listing_id","brand","price_czk","year","mileage_km",
    "fuel","gearbox","power_kw","body_type","image_urls",
    "category","condition","model_extracted","generation",
    "trim","engine_displacement_l","fuel_type","engine_code",
    "transmission"
]

for c in required_cols:
    if c not in df.columns:
        df[c] = ""

df_cleaned = df.copy()

# brand filter
brand_counts = df_cleaned["brand"].value_counts()
valid_brands = brand_counts[brand_counts >= 3000].index
df_cleaned = df_cleaned[df_cleaned["brand"].isin(valid_brands)].reset_index(drop=True)


# SAVE
df_cleaned.to_csv(CLEANED_FILE, index=False, encoding="utf-8-sig")

df.to_csv(
    LABELED_FILE,
    columns=required_cols,
    index=False,
    encoding="utf-8-sig"
)

print("Saved labeled:", LABELED_FILE)
print("Saved cleaned:", CLEANED_FILE)
print("Shape labeled:", df.shape)
print("Shape cleaned:", df_cleaned.shape)