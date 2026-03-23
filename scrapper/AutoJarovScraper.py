import requests
from bs4 import BeautifulSoup
import re
import json
import os
from datetime import datetime

def extract_engine_type(text: str):
    """
    Extract engine type from the car title, e.g., 1.9 TDI, 1.5 TSI.
    """
    if not text:
        return None
    text = text.replace(",", ".")
    match = re.search(r"\d\.\d\s*[A-Za-z]+", text)
    return match.group(0).upper() if match else None

def extract_transmission(car_soup):
    """
    Returns 'Automatic' or 'Manual'.
    """
    params_ul = car_soup.select_one("ul.parameters")
    if not params_ul:
        return None
    params = [li.get_text(strip=True).lower() for li in params_ul.select("li")]
    for param in params:
        if "automat" in param:
            return "Automatic"
    return "Manual"

def extract_fuel_type(car_soup):
    """
    Returns 'Petrol', 'Diesel', 'Hybrid', or None if not found.
    """
    params_ul = car_soup.select_one("ul.parameters")
    if not params_ul:
        return None
    params = [li.get_text(strip=True).lower() for li in params_ul.select("li")]
    for param in params:
        if "benzín" in param:
            return "Petrol"
        if "nafta" in param:
            return "Diesel"
        if "hybrid" in param:
            return "Hybrid"
    return None

class AutoJarovScraper:

    BASE_URL = "https://www.autojarov.cz/nabidka-vozu/"
    DOMAIN = "https://www.autojarov.cz"
    HEADERS = {"User-Agent": "Mozilla/5.0"}
    CURRENT_YEAR = datetime.now().year

    def get_type(self):
        return "autojarov"

    def _save_partial(self, records, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = []

        existing.extend(records)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def scrape_page(self, page, save_path="data/raw/cars.json"):
        url = self.BASE_URL + str(page)
        response = requests.get(url, headers=self.HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        list_container = soup.find("div", id="searchEngine-listVehicles-list")

        if not list_container:
            return []

        car_cards = list_container.select("a.box.vehicle-smallCard")

        page_records = []
        buffer = []

        for car in car_cards:
            try:
                h4 = car.find("h4")
                brand = h4.contents[0].strip()
                full_title = h4.find("b").get_text(strip=True)

                engine_type = extract_engine_type(full_title)
                model = full_title
                if engine_type:
                    model = model.replace(engine_type.replace(".", ","), "")
                model = re.sub(r"\s{2,}", " ", model).strip()

                price_raw = car.select_one(".prices b").get_text(strip=True)
                price = int(re.sub(r"[^\d]", "", price_raw))

                images = []
                for img in car.select("img"):
                    url = img.get("data-src") or img.get("src")
                    if url and url.startswith("/"):
                        url = self.DOMAIN + url
                    if url:
                        images.append(url)
                    if len(images) >= 5:
                        break

                transmission = extract_transmission(car)
                fuel_type = extract_fuel_type(car)

                record = {
                    "brand": brand,
                    "model": model,
                    "engine_type": engine_type,
                    "transmission": transmission,
                    "fuel_type": fuel_type,
                    "price": price,
                    "images": images[:5],
                    "source": "autojarov"
                }

                page_records.append(record)
                buffer.append(record)

                if len(buffer) >= 20:
                    self._save_partial(buffer, save_path)
                    buffer.clear()

            except Exception as e:
                print("Skipped car:", e)

        if buffer:
            self._save_partial(buffer, save_path)

        return page_records

