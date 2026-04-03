# car_title_parser.py

import re
import unicodedata
from typing import Optional, Dict, Any

import pandas as pd


# =========================
# NORMALIZATION
# =========================

def normalize_text(text: str) -> str:
    """
    Normalize raw listing title / model:
    - lowercase
    - remove accents
    - normalize decimal commas (1,4 -> 1.4)
    - remove noisy punctuation
    - fix common typos / aliases
    """
    if text is None or pd.isna(text):
        return ""

    text = str(text).lower()

    # remove accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # normalize engine decimals: 1,4 -> 1.4
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)

    # remove quotes
    text = text.replace('"', " ").replace("'", " ")

    # replace separators
    text = re.sub(r"[|/\\()\[\]{}:;]", " ", text)
    text = re.sub(r"[-_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # common typos / aliases
    replacements = {
        r"\bskoda\b": "",
        r"\bskoda\s+auto\b": "",
        r"\bskala\b": "scala",
        r"\bkodiag\b": "kodiaq",
        r"\bkodiak\b": "kodiaq",
        r"\bfabie\b": "fabia",
        r"\bfelicie\b": "felicia",
        r"\boctavie\b": "octavia",
        r"\bocta\b": "octavia",
        r"\bsuper\b": "superb",
        r"\bautomat\b": "automatic",
        r"\bman\b": "manual",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _combine_sources(title: Optional[str], model: Optional[str]) -> str:
    """
    Combine title + model into one parsing string.
    Prefer both because one field is often incomplete.
    """
    title_norm = normalize_text(title)
    model_norm = normalize_text(model)

    if title_norm and model_norm:
        return f"{model_norm} {title_norm}".strip()
    return title_norm or model_norm or ""


MODEL_PATTERNS = [
    ("Superb", [r"\bsuperb\b"]),
    ("Octavia", [r"\boctavia\b"]),
    ("Kodiaq", [r"\bkodiaq\b"]),
    ("Karoq", [r"\bkaroq\b"]),
    ("Kamiq", [r"\bkamiq\b"]),
    ("Scala", [r"\bscala\b"]),
    ("Fabia", [r"\bfabia\b"]),
    ("Rapid", [r"\brapid\b"]),
    ("Roomster", [r"\broomster\b"]),
    ("Yeti", [r"\byeti\b"]),
    ("Enyaq", [r"\benyaq\b"]),
    ("Citigo", [r"\bcitigo\b"]),
    ("Felicia", [r"\bfelicia\b"]),

    # mixed-brand fallback
    ("Passat", [r"\bpassat\b"]),
    ("Golf", [r"\bgolf\b"]),
    ("Tiguan", [r"\btiguan\b"]),
    ("Touran", [r"\btouran\b"]),
    ("Arteon", [r"\barteon\b"]),
    ("A4", [r"\ba4\b"]),
    ("A6", [r"\ba6\b"]),
    ("Q5", [r"\bq5\b"]),
    ("X3", [r"\bx3\b"]),
    ("X5", [r"\bx5\b"]),
]

BODY_PATTERNS = [
    ("Combi", [r"\bcombi\b", r"\bkombi\b"]),
    ("Sedan", [r"\bsedan\b"]),
    ("Spaceback", [r"\bspaceback\b"]),
    ("Liftback", [r"\bliftback\b", r"\blim\b", r"\blift\b"]),
    ("Scout", [r"\bscout\b"]),
    ("SUV", [r"\bsuv\b"]),
]

TRIM_PATTERNS = [
    ("RS", [r"\brs\b", r"\bvrs\b"]),
    ("Sportline", [r"\bsportline\b"]),
    ("L&K", [r"\bl\s*&\s*k\b", r"\blaurin\b"]),
    ("Monte Carlo", [r"\bmonte\s+carlo\b"]),
    ("Style", [r"\bstyle\b"]),
    ("Challenge", [r"\bchallenge\b"]),
    ("Scout", [r"\bscout\b"]),
]


# =========================
# HELPERS
# =========================

def _first_match(text: str, patterns) -> Optional[str]:
    for label, regexes in patterns:
        for rx in regexes:
            if re.search(rx, text, flags=re.IGNORECASE):
                return label
    return None


def _extract_generation(text: str, model: Optional[str]) -> Optional[str]:
    """
    Safer generation extraction.
    Avoid false positives from engine sizes like 1.4, 2.0, 1.9 TDI, etc.
    Prefer generation if it appears close to the model name.
    """
    if not model:
        return None

    temp = re.sub(r"\b\d+\.\d+\b", " ", text)
    model_base = model.split()[0].lower()

    proximity_patterns = [
        (rf"\b{model_base}\s+(iv|iii|ii|i)\b", {"iv": "IV", "iii": "III", "ii": "II", "i": "I"}),
        (rf"\b{model_base}\s+([1-4])\b", {"1": "I", "2": "II", "3": "III", "4": "IV"}),
        (rf"\b(iv|iii|ii|i)\s+{model_base}\b", {"iv": "IV", "iii": "III", "ii": "II", "i": "I"}),
        (rf"\b([1-4])\s+{model_base}\b", {"1": "I", "2": "II", "3": "III", "4": "IV"}),
    ]

    for pattern, mapping in proximity_patterns:
        match = re.search(pattern, temp, flags=re.IGNORECASE)
        if match:
            return mapping.get(match.group(1).lower())

    return None


# =========================
# ENGINE / TECH EXTRACTION
# =========================

def _extract_engine_displacement(text: str) -> Optional[float]:
    t = text.replace(",", ".")

    # standard: 2.0 tdi / 1.5 tsi / 1.9tdi
    match = re.search(
        r"\b(0\.\d|1\.\d|2\.\d|3\.\d|4\.\d)\s*(tdi|tsi|tfsi|fsi|mpi|dci|hdi|cdi|gdi|iv|etec|ecotsi)?\b",
        t
    )
    if match:
        try:
            return float(match.group(1))
        except:
            pass

    # dirty bazaar style: 16tdi -> 1.6, 12tsi -> 1.2, 20tdi -> 2.0
    match = re.search(r"\b(10|12|14|15|16|18|19|20|25|28|30|32|36|40)\s*(tdi|tsi|tfsi|fsi|mpi|hdi|cdi|gdi)\b", t)
    if match:
        raw = match.group(1)
        try:
            return float(f"{raw[0]}.{raw[1:]}")
        except:
            pass

    return None


def _extract_fuel_type(text: str) -> Optional[str]:
    t = text.lower()

    if re.search(r"\benyaq\b|\bev\b|\belectric\b", t):
        if re.search(r"\biv\b|\bphev\b|\bplugin\b|\bplug in\b", t):
            return "PHEV"
        return "EV"

    if re.search(r"\biv\b|\bphev\b|\bhybrid\b|\bplugin\b|\bplug in\b", t):
        return "PHEV"

    if re.search(r"\btdi\b|\bdci\b|\bhdi\b|\bcdi\b|\bcrdi\b", t):
        return "Diesel"

    if re.search(r"\btsi\b|\btfsi\b|\bfsi\b|\bgdi\b|\becotsi\b", t):
        return "Petrol"

    if re.search(r"\bmpi\b|\bhtp\b|\b16v\b", t):
        return "Petrol"

    return None


def _extract_engine_code(text: str) -> Optional[str]:
    t = text.lower()

    engine_codes = [
        "tdi", "tsi", "tfsi", "fsi", "mpi", "htp", "gdi",
        "dci", "hdi", "cdi", "crdi", "iv", "phev", "ev"
    ]

    for code in engine_codes:
        if re.search(rf"\b{re.escape(code)}\b", t):
            return code.upper()

    return None


def _extract_power_kw(text: str) -> Optional[int]:
    match = re.search(r"\b(\d{2,3})\s*k\s*w\b|\b(\d{2,3})\s*kw\b", text.lower())
    if match:
        value = match.group(1) or match.group(2)
        try:
            return int(value)
        except:
            return None
    return None


def _extract_horsepower_hp(text: str) -> Optional[int]:
    match = re.search(r"\b(\d{2,3})\s*(hp|ps)\b", text.lower())
    if match:
        try:
            return int(match.group(1))
        except:
            return None
    return None


def _extract_transmission(text: str) -> Optional[str]:
    t = text.lower()

    automatic_patterns = [
        r"\bdsg\b",
        r"\bautomatic\b",
        r"\bautomat\b",
        r"\bs tronic\b",
        r"\btiptronic\b",
        r"\bcvt\b",
        r"\bat\b"
    ]

    manual_patterns = [
        r"\bmanual\b",
        r"\bman\b",
        r"\bmt\b",
        r"\b6st\b",
        r"\b5st\b"
    ]

    for pattern in automatic_patterns:
        if re.search(pattern, t):
            return "Automatic"

    for pattern in manual_patterns:
        if re.search(pattern, t):
            return "Manual"

    # default assumption in listing data
    return "Manual"


def _extract_drivetrain(text: str) -> Optional[str]:
    t = text.lower()

    if re.search(r"\b4x4\b|\b4wd\b|\bawd\b", t):
        return "AWD"

    if re.search(r"\brwd\b", t):
        return "RWD"

    if re.search(r"\bfwd\b", t):
        return "FWD"

    return None


# =========================
# MAIN EXTRACTOR
# =========================

def extract_car_info(title: Optional[str], model: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract structured car data from BOTH title and model.

    Parameters:
        title: messy listing title
        model: optional cleaner model column

    Returns:
        dict with parsed fields
    """
    combined = _combine_sources(title, model)

    model_extracted = _first_match(combined, MODEL_PATTERNS)
    generation = _extract_generation(combined, model_extracted)
    body_type = _first_match(combined, BODY_PATTERNS)
    trim = _first_match(combined, TRIM_PATTERNS)

    engine_displacement_l = _extract_engine_displacement(combined)
    fuel_type = _extract_fuel_type(combined)
    engine_code = _extract_engine_code(combined)
    power_kw = _extract_power_kw(combined)
    horsepower_hp = _extract_horsepower_hp(combined)
    transmission = _extract_transmission(combined)
    drivetrain = _extract_drivetrain(combined)

    return {
        "normalized_source": combined,
        "model_extracted": model_extracted,
        "generation": generation,
        "body_type": body_type,
        "trim": trim,
        "engine_displacement_l": engine_displacement_l,
        "fuel_type": fuel_type,
        "engine_code": engine_code,
        "power_kw": power_kw,
        "horsepower_hp": horsepower_hp,
        "transmission": transmission,
        "drivetrain": drivetrain,
    }


# =========================
# PIPELINE FUNCTION
# =========================

def clean_car_columns(
    df: pd.DataFrame,
    title_col: str = "title",
    model_col: str = "model",
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Pipeline-ready cleaner:
    - uses BOTH title and model columns
    - appends parsed columns
    - can optionally overwrite existing parsed fields

    Parameters:
        df: input DataFrame
        title_col: column containing raw title
        model_col: column containing model text
        overwrite: if True, overwrite existing parsed columns

    Returns:
        cleaned DataFrame
    """
    result = df.copy()

    if title_col not in result.columns:
        result[title_col] = None

    if model_col not in result.columns:
        result[model_col] = None

    parsed = result.apply(
        lambda row: extract_car_info(
            title=row.get(title_col),
            model=row.get(model_col)
        ),
        axis=1
    ).apply(pd.Series)

    for col in parsed.columns:
        if col not in result.columns:
            result[col] = parsed[col]
        else:
            if overwrite:
                result[col] = parsed[col]
            else:
                result[col] = result[col].fillna(parsed[col])

    return result


if __name__ == "__main__":
    sample = pd.DataFrame({
        "title": [
            "SUPERB 3 1,4TSi",
            "2.0 TDi,147kW,4x4,DSG,L&K,LED,tažné",
            "Octavia 4 RS 2.0TDI 147kW DSG",
            "Rapid Spaceback 1.2 TSI",
            "Enyaq iV 60"
        ],
        "model": [
            "Superb",
            "Kodiaq",
            None,
            None,
            "Enyaq"
        ]
    })

    cleaned = clean_car_columns(sample, title_col="title", model_col="model")
    print(cleaned)