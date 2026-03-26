import re


def clean_text(text: str):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def parse_price(text: str):
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None


def parse_mileage(text: str):
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None

def extract_engine_type(text: str):
    """
    Extract engine type from title, e.g.:
    1.9 TDI
    2.0 TSI
    1.5 ECOBOOST
    """
    if not text:
        return None

    normalized = text.replace(",", ".")
    match = re.search(r"\d\.\d\s*[A-Za-z]+", normalized)

    return match.group(0).upper() if match else None