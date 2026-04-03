import re
import os
import csv
import time
import random
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://auto.bazos.cz/"
TARGET_COUNT = 200_0000
OUTPUT_CSV = "../data/bazos_cars_10k.csv"
SEEN_IDS_FILE = "../data/seen_ids.txt"
FAILED_URLS_FILE = "../data/failed_urls.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
}

# Good starting points for broad coverage
START_PAGES = [
    "https://auto.bazos.cz/skoda/",
    "https://auto.bazos.cz/volkswagen/",
    "https://auto.bazos.cz/bmw/",
    "https://auto.bazos.cz/audi/",
    "https://auto.bazos.cz/mercedes/",
    "https://auto.bazos.cz/ford/",
    "https://auto.bazos.cz/toyota/",
    "https://auto.bazos.cz/hyundai/",
    "https://auto.bazos.cz/renault/",
    "https://auto.bazos.cz/opel/",
    "https://auto.bazos.cz/peugeot/",
    "https://auto.bazos.cz/kia/",
    "https://auto.bazos.cz/mazda/",
    "https://auto.bazos.cz/honda/",
    "https://auto.bazos.cz/nissan/",
    "https://auto.bazos.cz/seat/",
    "https://auto.bazos.cz/citroen/",
    "https://auto.bazos.cz/fiat/",
    "https://auto.bazos.cz/suzuki/",
    "https://auto.bazos.cz/volvo/",
]

CSV_FIELDS = [
    "listing_id",
    "url",
    "title",
    "brand",
    "model",
    "price_czk",
    "year",
    "mileage_km",
    "fuel",
    "gearbox",
    "power_kw",
    "body_type",
    "location",
    "postal_code",
    "posted_date",
    "views",
    "phone",
    "description",
    "image_urls",
    "category"
]

session = requests.Session()
session.headers.update(HEADERS)


# -----------------------------
# Helpers
# -----------------------------
def sleep():
    time.sleep(random.uniform(0.01, 0.05))


def fetch(url):
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text


def parse_listing_id(url):
    m = re.search(r"/inzerat/(\d+)", url)
    return m.group(1) if m else None


def load_seen_ids():
    if not os.path.exists(SEEN_IDS_FILE):
        return set()

    with open(SEEN_IDS_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def append_seen_id(listing_id):
    with open(SEEN_IDS_FILE, "a", encoding="utf-8") as f:
        f.write(listing_id + "\n")


def append_failed_url(url):
    with open(FAILED_URLS_FILE, "a", encoding="utf-8") as f:
        f.write(url + "\n")


def init_csv():
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_row(row):
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def get_current_count(seen_ids):
    return len(seen_ids)


def clean_text(text):
    if not text:
        return None
    return re.sub(r"\s+", " ", text).strip()


def normalize_number(num_str):
    if not num_str:
        return None
    return re.sub(r"[^\d]", "", num_str)


# -----------------------------
# Listing page parsing
# -----------------------------
def extract_listing_links(list_page_html):
    soup = BeautifulSoup(list_page_html, "lxml")
    links = []

    for a in soup.select("a[href]"):
        href = a.get("href", "")
        full = urljoin(BASE_URL, href)

        if "/inzerat/" in full:
            links.append(full)

    return list(dict.fromkeys(links))


def extract_next_page(list_page_html, current_url):
    soup = BeautifulSoup(list_page_html, "lxml")

    # First try obvious next buttons
    for a in soup.select("a[href]"):
        text = a.get_text(" ", strip=True).lower()
        href = a.get("href", "")
        full = urljoin(current_url, href)

        if "další" in text or "dalsi" in text or "next" in text:
            return full

    # Fallback: Bazoš often uses pagination with offsets
    page_links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        full = urljoin(current_url, href)
        if full.startswith(current_url.split("?")[0]) and full != current_url:
            page_links.append(full)

    page_links = list(dict.fromkeys(page_links))

    # Heuristic: choose the first pagination-like URL that looks like next page
    for link in page_links:
        if re.search(r"/\d+/?$", link) or re.search(r"\?hledat=", link):
            return link

    return None


# -----------------------------
# Field extraction
# -----------------------------
def extract_price(text):
    m = re.search(r"(\d[\d\s]{2,})\s*Kč", text, re.IGNORECASE)
    return normalize_number(m.group(1)) if m else None


def extract_date(text):
    # [30.3. 2026]
    m = re.search(r"\[(\d{1,2}\.\d{1,2}\.\s*\d{4})\]", text)
    return clean_text(m.group(1)) if m else None


def extract_postal_code(text):
    m = re.search(r"\b(\d{3}\s?\d{2})\b", text)
    return m.group(1) if m else None


def extract_views(text):
    # often "123 x"
    m = re.search(r"\b(\d+)\s*x\b", text)
    return m.group(1) if m else None


def extract_phone(text):
    # ONLY if publicly visible in HTML
    candidates = re.findall(r"(\+?\d[\d\s]{7,}\d)", text)
    for c in candidates:
        digits = re.sub(r"\D", "", c)
        if 9 <= len(digits) <= 15:
            return clean_text(c)
    return None


def extract_year(text):
    # realistic car year range
    years = re.findall(r"\b(19[89]\d|20[0-2]\d|2030)\b", text)
    if years:
        # often the first valid one is usable
        return years[0]
    return None


def extract_mileage(text):
    patterns = [
        r"(\d[\d\s]{1,8})\s*km\b",
        r"najeto[:\s]+(\d[\d\s]{1,8})\b",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return normalize_number(m.group(1))
    return None


def extract_power_kw(text):
    m = re.search(r"(\d{2,4})\s*kW\b", text, re.IGNORECASE)
    return m.group(1) if m else None


def extract_fuel(text):
    fuels = [
        "benzín", "benzin", "nafta", "diesel", "hybrid",
        "elektro", "lpg", "cng", "plug-in hybrid"
    ]
    t = text.lower()
    for fuel in fuels:
        if fuel in t:
            return fuel
    return None


def extract_gearbox(text):
    gearboxes = [
        "manuální", "manual", "automat", "automatická", "dsg"
    ]
    t = text.lower()
    for g in gearboxes:
        if g in t:
            return g
    return None


def extract_body_type(text):
    body_types = [
        "sedan", "kombi", "hatchback", "suv", "cupe", "coupé",
        "liftback", "combi", "van", "mpv", "pick-up", "pickup",
        "kabriolet", "roadster", "terénní", "offroad"
    ]
    t = text.lower()
    for bt in body_types:
        if bt in t:
            return bt
    return None


def infer_brand_model(title, category):
    title_clean = clean_text(title) or ""
    parts = title_clean.split()

    brand = None
    model = None

    if parts:
        brand = parts[0]

    # Prefer category if it looks more reliable
    if category and category.lower() not in ["auto", "inzerat"]:
        brand = category.capitalize()

    if len(parts) >= 2:
        model = " ".join(parts[1:4])  # first few words after brand

    return brand, model


def extract_location(lines, postal_code):
    if not postal_code:
        return None

    for i, line in enumerate(lines):
        if postal_code in line:
            if i > 0:
                return clean_text(lines[i - 1])
            return clean_text(line)
    return None


def extract_description(soup):
    # Try to find the longest meaningful block
    candidates = []
    for tag in soup.find_all(["div", "p"]):
        txt = clean_text(tag.get_text(" ", strip=True))
        if txt and len(txt) > 80:
            candidates.append(txt)

    if not candidates:
        return None

    # Usually the longest one is the ad description
    best = max(candidates, key=len)
    return best[:8000]


def extract_images(soup, url):
    imgs = []

    for img in soup.select("img[src]"):
        src = img.get("src")
        if not src:
            continue

        full = urljoin(url, src)

        if any(ext in full.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            imgs.append(full)

    # dedupe
    imgs = list(dict.fromkeys(imgs))
    return imgs


# -----------------------------
# Detail page parsing
# -----------------------------
def parse_detail(html, url, category="auto"):
    soup = BeautifulSoup(html, "lxml")
    raw_text = soup.get_text("\n", strip=True)
    text = clean_text(raw_text)
    lines = [clean_text(x) for x in raw_text.splitlines() if clean_text(x)]

    listing_id = parse_listing_id(url)

    # Title
    title = None
    h1 = soup.find(["h1", "h2"])
    if h1:
        title = clean_text(h1.get_text(" ", strip=True))

    price_czk = extract_price(raw_text)
    posted_date = extract_date(raw_text)
    postal_code = extract_postal_code(raw_text)
    views = extract_views(raw_text)
    phone = extract_phone(raw_text)

    year = extract_year(raw_text)
    mileage_km = extract_mileage(raw_text)
    power_kw = extract_power_kw(raw_text)
    fuel = extract_fuel(raw_text)
    gearbox = extract_gearbox(raw_text)
    body_type = extract_body_type(raw_text)

    brand, model = infer_brand_model(title, category)
    location = extract_location(lines, postal_code)
    description = extract_description(soup)
    images = extract_images(soup, url)

    return {
        "listing_id": listing_id,
        "url": url,
        "title": title,
        "brand": brand,
        "model": model,
        "price_czk": price_czk,
        "year": year,
        "mileage_km": mileage_km,
        "fuel": fuel,
        "gearbox": gearbox,
        "power_kw": power_kw,
        "body_type": body_type,
        "location": location,
        "postal_code": postal_code,
        "posted_date": posted_date,
        "views": views,
        "phone": phone,
        "description": description,
        "image_urls": " | ".join(images),
        "category": category
    }


# -----------------------------
# Main scraping loop
# -----------------------------
def scrape():
    init_csv()
    seen_ids = load_seen_ids()

    print(f"Already scraped: {len(seen_ids)} listings")

    for start_url in START_PAGES:
        if len(seen_ids) >= TARGET_COUNT:
            break

        category = start_url.rstrip("/").split("/")[-1]
        current_url = start_url
        pages_scraped = 0

        print(f"\n=== CATEGORY: {category} ===")

        while current_url and len(seen_ids) < TARGET_COUNT and pages_scraped < 300:
            try:
                print(f"[LIST] {current_url}")
                html = fetch(current_url)
                sleep()

                listing_links = extract_listing_links(html)
                print(f"Found {len(listing_links)} listing links")

                if not listing_links:
                    break

                for link in listing_links:
                    if len(seen_ids) >= TARGET_COUNT:
                        break

                    listing_id = parse_listing_id(link)
                    if not listing_id:
                        continue
                    if listing_id in seen_ids:
                        continue

                    try:
                        print(f"  [DETAIL] {link}")
                        detail_html = fetch(link)
                        sleep()

                        data = parse_detail(detail_html, link, category=category)

                        if data["listing_id"]:
                            append_row(data)
                            append_seen_id(data["listing_id"])
                            seen_ids.add(data["listing_id"])

                            print(
                                f"    Saved #{data['listing_id']} | "
                                f"{data.get('brand')} {data.get('model')} | "
                                f"total={len(seen_ids)}"
                            )

                    except Exception as e:
                        print(f"    ERROR detail {link}: {e}")
                        append_failed_url(link)

                next_page = extract_next_page(html, current_url)
                if not next_page or next_page == current_url:
                    print("No next page found, moving to next category.")
                    break

                current_url = next_page
                pages_scraped += 1

            except Exception as e:
                print(f"ERROR list page {current_url}: {e}")
                break

    print(f"\nDone. Scraped {len(seen_ids)} listings.")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    scrape()