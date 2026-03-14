import pandas as pd
import re


def normalize_model(model: str) -> str:
    """
    Zjednoduší název modelu na základní model.
    """

    if not isinstance(model, str):
        return model

    model = model.strip()

    # remove engine info
    model = re.sub(r"\d\.\d\s*[A-Za-z]+", "", model)

    remove_words = [
        "TDI", "TSI", "FSI", "CDI", "HDI",
        "DSG", "Stronic", "S-tronic", "Tiptronic",
        "quattro", "4x4", "AWD",
        "Combi", "Avant", "Touring", "Variant",
        "Beach", "Style", "Selection", "Sportline"
    ]

    for word in remove_words:
        model = re.sub(rf"\b{word}\b", "", model, flags=re.IGNORECASE)

    model = re.sub(r"\b\d+(\.\d+)?\b", "", model)
    model = re.sub(r"\s+", " ", model)

    if model:
        model = model.split()[0]

    return model.strip()


def normalize_brand(brand: str):
    if not isinstance(brand, str):
        return brand

    if "Volkswagen Užitkové vozy" in brand:
        return "Volkswagen"

    return brand

def normalize_fuel(fuel: str):
    if not isinstance(fuel, str):
        return fuel

    fuel = fuel.lower()

    if "nafta" in fuel or "diesel" in fuel:
        return "Diesel"

    if "benz" in fuel:
        return "Petrol"

    if "hybrid" in fuel:
        return "Hybrid"

    if "elektro" in fuel or "electric" in fuel:
        return "Electric"

    return fuel.capitalize()


def normalize_transmission(transmission: str):

    if not isinstance(transmission, str):
        return "Manual"

    transmission = transmission.lower()

    if any(x in transmission for x in ["automat", "dsg", "stronic", "tiptronic"]):
        return "Automatic"

    return "Manual"


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:

    print("Původní velikost datasetu:", df.shape)

    # km NaN → 0
    df["km"] = df["km"].fillna(0)

    # normalize columns
    df["model"] = df["model"].apply(normalize_model)
    df["fuel"] = df["fuel"].apply(normalize_fuel)
    df["transmission"] = df["transmission"].apply(normalize_transmission)

    # remove invalid data
    df = df[df["km"] >= 0]
    df = df[df["price"] > 0]
    df = df[df["year"] > 1990]

    df = df.drop_duplicates()

    df = df.dropna(subset=["brand", "model", "price", "year"])

    print("Velikost po základním čištění:", df.shape)

    return df


def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    cleaned_df = df[
        (df[column] >= lower_bound) &
        (df[column] <= upper_bound)
    ]

    print(f"Outliery odstraněny ze sloupce {column}")
    print("Velikost po odstranění outlierů:", cleaned_df.shape)

    return cleaned_df


def full_cleaning_pipeline(df: pd.DataFrame, output_path: str = "../data/cars_cleaned.csv") -> pd.DataFrame:

    df = basic_cleaning(df)

    ##df = remove_outliers(df, "price")
    ##df = remove_outliers(df, "km")

    df = df.drop_duplicates()

    print("Finální velikost datasetu:", df.shape)

    # SAVE CLEANED DATASET
    df.to_csv(output_path, index=False)

    print(f"Dataset uložen do: {output_path}")

    return df


df = pd.read_csv("cars_dataset.csv")

cleaned = full_cleaning_pipeline(df)