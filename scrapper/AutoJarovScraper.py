import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

from scrapper.utils.image_extractor import extract_gallery_images, normalize_url
from scrapper.utils.text_utils import clean_text, parse_price, parse_mileage, extract_engine_type
from scrapper.utils.fetch_utils import fetch_url


class AutoJarovScraper:
    BASE_URL = "https://www.autojarov.cz/nabidka-vozu/"
    DOMAIN = "https://www.autojarov.cz"
    CURRENT_YEAR = datetime.now().year

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def get_type(self):
        return "autojarov"

    def _extract_detail_data(self, detail_url: str):
        response = fetch_url(
            detail_url,
            headers=self.HEADERS,
            timeout=20,
            retries=3,
            backoff=2,
            min_delay=0.2,
            max_delay=0.7
        )
        if not response:
            print(f"[AutoJarov] Failed detail page {detail_url}")
            return {
                "images": [],
                "fuel_type": None,
                "transmission": None,
                "mileage_km": None,
                "price": None,
                "stock_id": None,
                "location": None
            }

        soup = BeautifulSoup(response.text, "html.parser")

        # -------------------------
        # Images
        # -------------------------
        images = extract_gallery_images(soup, self.DOMAIN, max_images=20)

        # -------------------------
        # Vehicle top info
        # -------------------------
        fuel_type = None
        transmission = None
        mileage_km = None
        stock_id = None
        location = None

        top_info_items = soup.select("ul.vehicle-top-info li")

        for li in top_info_items:
            text = clean_text(li.get_text(" ", strip=True))
            if not text:
                continue

            lower = text.lower()

            stock_match = re.search(r"EČ:\s*(\d+)", text)
            if stock_match:
                stock_id = stock_match.group(1)

            if "benzín" in lower:
                fuel_type = "Petrol"
            elif "nafta" in lower:
                fuel_type = "Diesel"
            elif "hybrid" in lower:
                fuel_type = "Hybrid"
            elif "elektro" in lower or "electric" in lower:
                fuel_type = "Electric"

            if "automatická převodovka" in lower or "automat" in lower:
                transmission = "Automatic"
            elif "manuální převodovka" in lower or "manuál" in lower:
                transmission = "Manual"

            if "km" in lower:
                mileage_km = parse_mileage(text)

            if "skladem" in lower:
                location_span = li.select_one("span")
                if location_span:
                    location = clean_text(location_span.get_text(strip=True))

        # -------------------------
        # Price
        # -------------------------
        price = None
        price_el = soup.select_one("#infoPrices .price span")
        if price_el:
            price = parse_price(price_el.get_text(strip=True))

        return {
            "images": images,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "mileage_km": mileage_km,
            "price": price,
            "stock_id": stock_id,
            "location": location
        }

    def scrape_page(self, page):
        url = self.BASE_URL + str(page)

        response = self.session.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        list_container = soup.find("div", id="searchEngine-listVehicles-list")

        if not list_container:
            return []

        car_cards = list_container.select("a.box.vehicle-smallCard")
        page_records = []

        for car in car_cards:
            try:
                h4 = car.find("h4")
                if not h4:
                    continue

                # Brand + title
                brand_raw = h4.contents[0].strip() if h4.contents else ""
                brand = clean_text(brand_raw)

                bold = h4.find("b")
                full_title = clean_text(
                    bold.get_text(strip=True) if bold else h4.get_text(" ", strip=True)
                )

                engine_type = extract_engine_type(full_title)
                model = full_title

                if engine_type:
                    model = model.replace(engine_type.replace(".", ","), "")
                    model = model.replace(engine_type.replace(",", "."), "")

                model = clean_text(model)

                # Detail URL
                detail_href = car.get("href")
                detail_url = normalize_url(self.DOMAIN, detail_href)
                if not detail_url:
                    continue

                # Detail data
                detail_data = self._extract_detail_data(detail_url)

                record = {
                    "brand": brand,
                    "model": model,
                    "engine_type": engine_type,
                    "transmission": detail_data["transmission"],
                    "fuel_type": detail_data["fuel_type"],
                    "mileage_km": detail_data["mileage_km"],
                    "price": detail_data["price"],
                    "images": detail_data["images"],
                    "url": detail_url,
                    "source": "autojarov"
                }

                page_records.append(record)

            except Exception as e:
                print("Skipped car:", e)

        return page_records