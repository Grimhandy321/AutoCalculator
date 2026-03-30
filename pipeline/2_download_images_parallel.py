import os
import re
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_FILE = "../data/bazos_cars_labeled.csv"
OUTPUT_DIR = "../data/car_images"
MAX_WORKERS = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

def safe_filename(text):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(text))

def download_one(row):
    listing_id = str(row["listing_id"]).strip()
    image_urls = str(row.get("image_urls", "")).strip()

    if not image_urls or image_urls == "nan":
        return f"SKIP {listing_id}: no image"

    first_image = image_urls.split(" | ")[0].strip()
    if not first_image.startswith("http"):
        return f"SKIP {listing_id}: invalid url"

    ext = ".jpg"
    for e in [".jpg", ".jpeg", ".png", ".webp"]:
        if e in first_image.lower():
            ext = e
            break

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

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_one, row) for _, row in df.iterrows()]
    for future in as_completed(futures):
        print(future.result())