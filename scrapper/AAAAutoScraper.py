import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
import requests

from scrapper.utils.fetch_utils import fetch_url
from scrapper.utils.image_extractor import extract_gallery_images, is_valid_car_image, normalize_url
from scrapper.utils.text_utils import clean_text, parse_price


def extract_engine_type(text: str):
    if not text:
        return None
    text = text.replace(",", ".")
    match = re.search(
        r"\b\d\.\d\s*(TSI|TDI|MPI|HDI|CDI|GDI|ECOBOOST|V\d)\b",
        text,
        re.IGNORECASE
    )
    return match.group(0).upper() if match else None


def extract_transmission(car_soup):
    """
    Extract transmission type from listing card tags.
    """
    tags_ul = car_soup.select_one("ul.columnsTags")
    if not tags_ul:
        return None

    tags = [li.get_text(strip=True).lower() for li in tags_ul.select("li.tag")]

    for tag in tags:
        if "automat" in tag:
            return "Automatic"
        if "manuál" in tag or "manual" in tag:
            return "Manual"

    return None


def extract_fuel_type(car_soup):
    """
    Try to extract fuel type from card tags.
    """
    tags_ul = car_soup.select_one("ul.columnsTags")
    if not tags_ul:
        return None

    tags = [li.get_text(strip=True).lower() for li in tags_ul.select("li.tag")]

    for tag in tags:
        if "benz" in tag:
            return "Petrol"
        if "nafta" in tag or "diesel" in tag:
            return "Diesel"
        if "hybrid" in tag:
            return "Hybrid"
        if "elektro" in tag or "electric" in tag:
            return "Electric"

    return None


class AAAAutoScraper:
    BASE_URL = "https://www.aaaauto.cz/ojete-vozy/#!&page="
    DOMAIN = "https://www.aaaauto.cz"
    HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "cs-CZ,cs;q=0.9,en;q=0.8"
    }

    DETAIL_WORKERS = 2

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def get_type(self):
        return "aaaauto"

    def _extract_detail_data(self, detail_url: str):
        """
        Scrape car detail page for better image set and extra metadata.
        """
        time.sleep(random.uniform(0.3, 1.0))

        response = fetch_url(
            detail_url,
            headers=self.HEADERS,
            timeout=10,
            retries=3,
            backoff=2,
            session=self.session
        )

        if not response:
            print(f"[AAAAuto] Failed detail page permanently: {detail_url}")
            return {
                "images": [],
                "mileage_km": None,
                "fuel_type": None,
                "transmission": None
            }

        soup = BeautifulSoup(response.text, "html.parser")

        # Generic image extraction
        images = extract_gallery_images(soup, self.DOMAIN, max_images=20)

        # Fallback: if AAA has specific gallery anchors / sliders, catch them too
        if not images:
            image_candidates = []

            for a in soup.select("a[href]"):
                href = a.get("href")
                if href:
                    full = normalize_url(href, self.DOMAIN)
                    if full and is_valid_car_image(full):
                        image_candidates.append(full)

            for img in soup.select("img"):
                src = img.get("data-src") or img.get("src")
                if src:
                    full = normalize_url(src, self.DOMAIN)
                    if full and is_valid_car_image(full):
                        image_candidates.append(full)

            seen = set()
            images = []
            for img in image_candidates:
                if img not in seen:
                    seen.add(img)
                    images.append(img)

        mileage_km = None
        fuel_type = None
        transmission = None

        # Try to extract from detail page text
        all_text = clean_text(soup.get_text(" ", strip=True)).lower()

        mileage_match = re.search(r"(\d[\d\s]{1,12})\s*km", all_text)
        if mileage_match:
            mileage_km = int(re.sub(r"[^\d]", "", mileage_match.group(1)))

        if "benzín" in all_text or "benzin" in all_text:
            fuel_type = "Petrol"
        elif "nafta" in all_text or "diesel" in all_text:
            fuel_type = "Diesel"
        elif "hybrid" in all_text:
            fuel_type = "Hybrid"
        elif "elektro" in all_text or "electric" in all_text:
            fuel_type = "Electric"

        if "automatická převodovka" in all_text or "automat" in all_text:
            transmission = "Automatic"
        elif "manuální převodovka" in all_text or "manuál" in all_text or "manual" in all_text:
            transmission = "Manual"

        return {
            "images": images[:20],
            "mileage_km": mileage_km,
            "fuel_type": fuel_type,
            "transmission": transmission
        }

    def _scrape_car_card(self, car):
        try:
            title = car.select_one("h2 a")
            if not title:
                return None

            title_text = clean_text(title.get_text(" ", strip=True))
            detail_href = title.get("href")
            detail_url = normalize_url(detail_href, self.DOMAIN) if detail_href else None

            parts = title_text.split()
            brand = parts[0] if parts else None

            year_match = re.search(r"\b(19|20)\d{2}\b", title_text)
            year = int(year_match.group()) if year_match else None

            engine_type = extract_engine_type(title_text)

            model = title_text
            if brand:
                model = re.sub(rf"^{re.escape(brand)}\s*", "", model, flags=re.IGNORECASE)
            if engine_type:
                model = re.sub(re.escape(engine_type), "", model, flags=re.IGNORECASE)
            if year:
                model = model.replace(str(year), "")
            model = re.sub(r"\s{2,}", " ", model).strip(" ,-")

            price_tag = car.select_one(".carPrice h3")
            price = parse_price(price_tag.get_text()) if price_tag else None

            # fallback listing images
            fallback_images = []
            for img in car.select("img"):
                src = img.get("data-src") or img.get("src")
                if src:
                    full = normalize_url(src, self.DOMAIN)
                    if full and is_valid_car_image(full):
                        fallback_images.append(full)

            transmission = extract_transmission(car)
            fuel_type = extract_fuel_type(car)

            detail_data = {
                "images": fallback_images[:5],
                "mileage_km": None,
                "fuel_type": fuel_type,
                "transmission": transmission
            }

            if detail_url:
                detail_data = self._extract_detail_data(detail_url)

            record = {
                "brand": brand,
                "model": model,
                "engine_type": engine_type,
                "transmission": detail_data.get("transmission") or transmission,
                "fuel_type": detail_data.get("fuel_type") or fuel_type,
                "price": price,
                "year": year,
                "mileage_km": detail_data.get("mileage_km"),
                "images": detail_data.get("images") or fallback_images[:5],
                "detail_url": detail_url,
                "source": "aaaauto"
            }

            return record

        except Exception as e:
            print("Skipped AAA car:", e)
            return None

    def scrape_page(self, page):
        url = self.BASE_URL + str(page)

        response = fetch_url(
            url,
            headers=self.HEADERS,
            timeout=10,
            retries=3,
            backoff=2,
            session=self.session
        )

        if not response:
            print(f"[AAAAuto] Failed page {page}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        container = soup.find("div", class_="carsGrid")

        if not container:
            return []

        cards = container.select("div.card.box")
        if not cards:
            return []

        page_records = []

        with ThreadPoolExecutor(max_workers=self.DETAIL_WORKERS) as executor:
            futures = [executor.submit(self._scrape_car_card, car) for car in cards]

            for future in as_completed(futures):
                record = future.result()
                if record:
                    page_records.append(record)

        return page_records