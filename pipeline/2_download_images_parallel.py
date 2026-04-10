import os
import threading
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.image_utils import (
    safe_filename,
    get_first_valid_image,
    get_extension_from_url,
    load_image_from_bytes,
    normalize_image,
    save_augmented_versions,
)

CSV_FILE = "../data/bazos_cars_labeled.csv"
OUTPUT_DIR = "../data/car_images"
AUGMENT_DIR = "../data/car_images_augmented"
PROGRESS_FILE = "../data/download_progress.csv"
MAX_WORKERS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUGMENT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)
df["listing_id"] = df["listing_id"].astype(str).str.strip()

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

lock = threading.Lock()


# -----------------------------
# Load previous progress
# -----------------------------
if os.path.exists(PROGRESS_FILE):
    progress_df = pd.read_csv(PROGRESS_FILE)
    processed_ids = set(progress_df["listing_id"].astype(str).str.strip())
    print(f"Loaded progress: {len(processed_ids)} already processed")
else:
    progress_df = pd.DataFrame(columns=["listing_id", "status", "message"])
    processed_ids = set()
    print("No previous progress file found. Starting fresh.")


# -----------------------------
# Save progress safely
# -----------------------------
def append_progress(listing_id: str, status: str, message: str):
    global progress_df

    row = pd.DataFrame([{
        "listing_id": listing_id,
        "status": status,
        "message": message
    }])

    with lock:
        progress_df = pd.concat([progress_df, row], ignore_index=True)
        row.to_csv(PROGRESS_FILE, mode="a", header=not os.path.exists(PROGRESS_FILE), index=False)


# -----------------------------
# Check if listing is already fully done
# -----------------------------
def is_fully_processed(base_name: str, ext: str) -> bool:
    original_path = os.path.join(OUTPUT_DIR, f"{base_name}{ext}")
    aug_original = os.path.join(AUGMENT_DIR, f"{base_name}_original.jpg")
    return os.path.exists(original_path) and os.path.exists(aug_original)


# -----------------------------
# Download one row
# -----------------------------
def download_one(row):
    listing_id = str(row["listing_id"]).strip()

    if listing_id in processed_ids:
        return f"SKIP {listing_id}: already logged"

    image_urls = str(row.get("image_urls", "")).strip()
    first_image = get_first_valid_image(image_urls)

    if not first_image:
        append_progress(listing_id, "skip", "no valid image")
        return f"SKIP {listing_id}: no valid image"

    ext = get_extension_from_url(first_image)
    base_name = safe_filename(listing_id)

    out_path = os.path.join(OUTPUT_DIR, f"{base_name}{ext}")

    # Resume-safe skip
    if is_fully_processed(base_name, ext):
        append_progress(listing_id, "done", "already exists")
        return f"EXISTS {listing_id}"

    try:
        r = session.get(first_image, timeout=20)
        r.raise_for_status()

        # Save original if missing
        if not os.path.exists(out_path):
            with open(out_path, "wb") as f:
                f.write(r.content)

        # Rebuild augmentations if missing / partial
        img = load_image_from_bytes(r.content)
        img = normalize_image(img)
        save_augmented_versions(img, base_name, AUGMENT_DIR)

        append_progress(listing_id, "done", "downloaded and augmented")
        return f"DOWNLOADED + AUGMENTED {listing_id}"

    except Exception as e:
        append_progress(listing_id, "failed", str(e))
        return f"FAILED {listing_id}: {e}"


# -----------------------------
# Main
# -----------------------------
def main():
    remaining_df = df[~df["listing_id"].isin(processed_ids)].copy()
    print(f"Remaining rows to process: {len(remaining_df)}")

    if len(remaining_df) == 0:
        print("Everything already processed.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_one, row) for _, row in remaining_df.iterrows()]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()