import os
import re
import requests
import pandas as pd
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_FILE = "../data/bazos_cars_labeled.csv"
OUTPUT_DIR = "../data/car_images"
MAX_WORKERS = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


# BLOCKED / INVALID IMAGES
BLOCKED_URLS = {
    "https://www.jasminka.cz/images/v/lecenizv.jpg"
}

BLOCKED_PATTERNS = [
    r"lecenizv\.jpg"
]


def safe_filename(text):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(text))


def is_blocked_image(url: str) -> bool:
    if not url:
        return True

    url_clean = url.strip().lower()

    if url_clean in {u.lower() for u in BLOCKED_URLS}:
        return True

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, url_clean, flags=re.IGNORECASE):
            return True

    return False


def get_first_valid_image(image_urls: str) -> Optional[str]:
    """
    From 'url1 | url2 | url3' return first valid non-blocked image.
    """
    if not image_urls or image_urls == "nan":
        return None

    urls = [u.strip() for u in image_urls.split(" | ") if u.strip()]

    for url in urls:
        if not url.startswith("http"):
            continue
        if is_blocked_image(url):
            continue
        return url

    return None


def get_extension_from_url(url: str) -> str:
    url_lower = url.lower()

    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        if ext in url_lower:
            return ext

    return ".jpg"


def download_one(row):
    listing_id = str(row["listing_id"]).strip()
    image_urls = str(row.get("image_urls", "")).strip()

    first_image = get_first_valid_image(image_urls)

    if not first_image:
        return f"SKIP {listing_id}: no valid image"

    ext = get_extension_from_url(first_image)
    out_path = os.path.join(OUTPUT_DIR, f"{safe_filename(listing_id)}{ext}")

    if os.path.exists(out_path):
        return f"EXISTS {listing_id}"

    try:
        r = session.get(first_image, timeout=20)
        r.raise_for_status()

        with open(out_path, "wb") as f:
            f.write(r.content)

        return f"DOWNLOADED {listing_id}"
    except Exception as e:
        return f"FAILED {listing_id}: {e}"


def main():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_one, row) for _, row in df.iterrows()]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()