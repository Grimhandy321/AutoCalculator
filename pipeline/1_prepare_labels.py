import re
import pandas as pd

from pipeline.utils.car_title_parser import clean_car_columns

CSV_FILE = "../data/bazos_cars_10k.csv"
OUTPUT_FILE = "../data/bazos_cars_labeled.csv"
BAD_IMAGE_URL = "https://www.jasminka.cz/images/v/lecenizv.jpg"

df = pd.read_csv(CSV_FILE,low_memory=False)


for col in ["title", "description", "brand", "model"]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str)

df["price_czk"] = pd.to_numeric(df.get("price_czk", None), errors="coerce")
df["year"] = pd.to_numeric(df.get("year", None), errors="coerce")
df["mileage_km"] = pd.to_numeric(df.get("mileage_km", None), errors="coerce")
df["power_kw"] = pd.to_numeric(df.get("power_kw", None), errors="coerce")

def clean_text(s):
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

df["brand"] = df["brand"].apply(clean_text)
df["model"] = df["model"].apply(clean_text)

# fallback from title
def infer_brand_model(row):
    brand = row["brand"]
    model = row["model"]
    title = row["title"]

    parts = title.split()

    if not brand and len(parts) >= 1:
        brand = parts[0]
    if (not model or model.lower() == brand.lower()) and len(parts) >= 2:
        model = " ".join(parts[1:3])

    return pd.Series([brand, model])

df[["brand", "model"]] = df.apply(infer_brand_model, axis=1)

excellent_kw = [
    "top stav", "perfektní", "jako nové", "jako nova", "bez investic",
    "garážované", "garazovane", "nehavarované", "nehavarovane",
    "servisní knížka", "servisni knizka"
]

good_kw = [
    "zachovalé", "zachovale", "dobrý stav", "dobry stav",
    "udržované", "udrzovane", "pravidelný servis", "pravidelny servis"
]

fair_kw = [
    "kosmetické vady", "kosmeticke vady", "horší stav", "horsi stav",
    "menší opravy", "mensi opravy", "opotřebení", "opotrebeni"
]

poor_kw = [
    "na opravu", "nepojízdné", "nepojizdne", "bourané", "bourane",
    "havárie", "havarie", "poškozené", "poskozene", "na díly", "na dily"
]

def infer_condition(text):
    t = text.lower()

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
        if kw in t:
            return "good"

    return "good"

df["condition"] = (df["title"] + " " + df["description"]).apply(infer_condition)


df = df.dropna(subset=["listing_id", "price_czk"])

df = df[df["price_czk"] > 10000]
df = df[df["price_czk"] < 5000000]

df = clean_car_columns(df, title_col="title", model_col="model")
df["image_urls"] = df["image_urls"].str.replace(BAD_IMAGE_URL, "", regex=False)


df.to_csv(OUTPUT_FILE,columns=['listing_id','brand','brand','price_czk','year','mileage_km','fuel','gearbox','power_kw','body_type','image_urls','category','condition','model_extracted','generation','trim','engine_displacement_l','fuel_type','engine_code','transmission',
],  index=False, encoding="utf-8-sig")
print(f"Saved labeled data: {OUTPUT_FILE}")
print("Shape:", df.shape)
print(df[["brand", "model_extracted", "condition"]].head())