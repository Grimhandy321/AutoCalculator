import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

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


class AutoJarovScraper(BaseScraper):

    BASE_URL = "https://www.autojarov.cz/nabidka-vozu/"
    HEADERS = {"User-Agent": "Mozilla/5.0"}

    CURRENT_YEAR = datetime.now().year

    def get_type(self):
        return "autojarov"

    def scrape_page(self, page):

        url = self.BASE_URL + str(page)

        response = requests.get(url, headers=self.HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        list_container = soup.find("div", id="searchEngine-listVehicles-list")

        if not list_container:
            return []

        car_cards = list_container.select("a.box.vehicle-smallCard")

        if not car_cards:
            return []

        page_records = []

        for car in car_cards:

            try:

                h4 = car.find("h4")

                brand = h4.contents[0].strip()

                full_title = h4.find("b").get_text(strip=True)

                # Example:
                # OCTAVIA TOP SELECTION 2,0 TDI DSG 110 kW

                engine_type = extract_engine_type(full_title)

                model = full_title

                if engine_type:
                    model = model.replace(engine_type.replace(".", ","), "")

                model = re.sub(r"\s{2,}", " ", model).strip()

                params = car.select("ul.parameters li")

                transmission = "Manual"
                fuel = None
                km = None
                year = None

                if len(params) >= 1:
                    transmission = params[0].get_text(strip=True)

                if len(params) >= 2:
                    fuel = params[1].get_text(strip=True)

                for p in params:

                    text = p.get_text(strip=True)

                    if "km" in text:
                        km = int(re.sub(r"[^\d]", "", text))

                    if "/" in text:
                        year = int(text.split("/")[-1])

                # Detect automatic transmission
                if transmission:
                    t = transmission.lower()

                    if any(x in t for x in ["automat", "dsg", "tiptronic"]):
                        transmission = "Automatic"
                    else:
                        transmission = "Manual"

                is_new = "nový vůz" in h4.get_text().lower()

                if is_new:
                    year = self.CURRENT_YEAR
                    km = 0

                price_raw = car.select_one(".prices b").get_text(strip=True)

                price = int(re.sub(r"[^\d]", "", price_raw))

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
                print("Skipped one car:", e)

        return page_records