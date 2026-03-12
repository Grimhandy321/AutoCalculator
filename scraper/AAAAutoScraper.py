import requests
from bs4 import BeautifulSoup
import re

from base_scraper import BaseScraper


def extract_engine_type(text: str):
    """
    Extract engine like '2.0 TDI', '1.5 TSI'
    """

    if not text:
        return None

    text = text.replace(",", ".")

    match = re.search(r"\d\.\d\s*[A-Za-z]+", text)

    if match:
        return match.group(0).upper()

    return None


class AAAAutoScraper(BaseScraper):

    BASE_URL = "https://www.aaaauto.cz/cz/inzerce/osobni-vozy/?page="
    HEADERS = {
        "User-Agent": "Mozilla/5.0"
    }

    def get_type(self):
        return "aaaauto"

    def scrape_page(self, page):

        url = self.BASE_URL + str(page)

        response = requests.get(url, headers=self.HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        container = soup.find("div", class_="carsGrid")

        if not container:
            return []

        cards = container.select("div.card.box")

        if not cards:
            return []

        page_records = []

        for car in cards:

            try:

                title = car.select_one("h2 a")
                title_text = title.get_text(" ", strip=True)

                # Example:
                # Škoda Superb 2.0 TDI 140kW 2019

                parts = title_text.split()

                brand = parts[0]

                # Extract year
                year_match = re.search(r"\b(19|20)\d{2}\b", title_text)
                year = int(year_match.group()) if year_match else None

                engine_type = extract_engine_type(title_text)

                model = title_text.replace(brand, "")

                if engine_type:
                    model = model.replace(engine_type, "")

                if year:
                    model = model.replace(str(year), "")

                model = re.sub(r"\s{2,}", " ", model).strip()

                tags = car.select("ul.columnsTags li")

                km = None
                fuel = None
                transmission = "Manual"

                for tag in tags:

                    text = tag.get_text(strip=True).lower()

                    if "km" in text:
                        km = int(re.sub(r"[^\d]", "", text))

                    elif any(x in text for x in ["diesel", "nafta"]):
                        fuel = "Diesel"

                    elif any(x in text for x in ["benz", "petrol"]):
                        fuel = "Petrol"

                    elif any(x in text for x in ["automat", "dsg", "tiptronic"]):
                        transmission = "Automatic"

                price_tag = car.select_one(".carPrice h3")

                price = None

                if price_tag:
                    price = int(re.sub(r"[^\d]", "", price_tag.get_text()))

                page_records.append({
                    "brand": brand,
                    "model": model,
                    "engine_type": engine_type,
                    "fuel": fuel,
                    "transmission": transmission,
                    "km": km,
                    "price": price,
                    "year": year
                })

            except Exception as e:
                print("Skipped AAA car:", e)

        return page_records