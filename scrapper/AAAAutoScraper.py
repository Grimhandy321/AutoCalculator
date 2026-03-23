import requests
from bs4 import BeautifulSoup
import re
import json
import os




def extract_engine_type(text: str):
    if not text:
        return None

    text = text.replace(",", ".")

    # matches: 2.0 TDI, 1.5 TSI, 3.0 V6, etc.
    match = re.search(r"\b\d\.\d\s*(TSI|TDI|MPI|HDI|CDI|GDI|ECOBOOST|V\d)\b", text, re.IGNORECASE)

    if match:
        return match.group(0).upper()

    return None

class AAAAutoScraper():

    BASE_URL = "https://www.aaaauto.cz/ojete-vozy/#!&page="
    DOMAIN = "https://www.aaaauto.cz"
    HEADERS = {"User-Agent": "Mozilla/5.0"}

    def get_type(self):
        return "aaaauto"

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
        container = soup.find("div", class_="carsGrid")

        if not container:
            return []

        cards = container.select("div.card.box")
        page_records = []
        buffer = []

        for car in cards:
            try:
                title = car.select_one("h2 a")
                title_text = title.get_text(" ", strip=True)

                parts = title_text.split()
                brand = parts[0]

                year_match = re.search(r"\b(19|20)\d{2}\b", title_text)
                year = int(year_match.group()) if year_match else None

                engine_type = extract_engine_type(title_text)

                model = title_text.replace(brand, "")
                if engine_type:
                    model = model.replace(engine_type, "")
                if year:
                    model = model.replace(str(year), "")
                model = re.sub(r"\s{2,}", " ", model).strip()

                price_tag = car.select_one(".carPrice h3")
                price = int(re.sub(r"[^\d]", "", price_tag.get_text())) if price_tag else None

                images = []
                for img in car.select("img"):
                    url = img.get("data-src") or img.get("src")
                    if url and url.startswith("/"):
                        url = self.DOMAIN + url
                    if url:
                        images.append(url)
                    if len(images) >= 5:
                        break

                record = {
                    "brand": brand,
                    "model": model,
                    "engine_type": engine_type,
                    "price": price,
                    "year": year,
                    "images": images[:5],
                    "source": "aaaauto"
                }

                page_records.append(record)
                buffer.append(record)

                if len(buffer) >= 20:
                    self._save_partial(buffer, save_path)
                    buffer.clear()

            except Exception as e:
                print("Skipped AAA car:", e)

        if buffer:
            self._save_partial(buffer, save_path)

        return page_records
