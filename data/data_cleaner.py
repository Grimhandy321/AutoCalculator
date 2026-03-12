import pandas as pd


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provede základní čištění datasetu.

    Kroky:
    - odstranění duplicit
    - odstranění záporných km
    - odstranění nulových / záporných cen
    - odstranění starých vozidel (rok <= 1990)
    - odstranění NaN hodnot

    Parameters:
        df (pd.DataFrame): vstupní dataset

    Returns:
        pd.DataFrame: vyčištěný dataset
    """

    print("Původní velikost datasetu:", df.shape)

    df = df.drop_duplicates()

    df = df[df["km"] >= 0]
    df = df[df["price"] > 0]
    df = df[df["year"] > 1990]

    df = df.dropna()

    print("Velikost po čištění:", df.shape)

    return df


def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Odstraní outliery
    Parameters:
        df (pd.DataFrame)
        column (str): název sloupce

    Returns:
        pd.DataFrame
    """

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


def full_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = basic_cleaning(df)
    df = remove_outliers(df, "price")
    df = df.drop_duplicates()
    df = df.dropna()

    return df
