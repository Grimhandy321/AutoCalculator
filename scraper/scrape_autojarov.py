import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import os
from datetime import datetime

BASE_URL = "https://www.autojarov.cz/nabidka-vozu/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

TARGET_RECORDS = 2000
DELAY = 0.1
CURRENT_YEAR = datetime.now().year

OUTPUT_FILE = "../data/autojarov_cars.csv"

if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    cars = existing_df.to_dict("records")
    print(f"existing file with {len(cars)} records.")
else:
    cars = []
    print("No existing file")

page = 1


def save_progress():
    df = pd.DataFrame(cars)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Progress saved. Total records: {len(df)}")


try:
    while len(cars) < TARGET_RECORDS:
        print(f"Scraping page {page}...")

        response = requests.get(BASE_URL + str(page), headers=HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        list_container = soup.find("div", id="searchEngine-listVehicles-list")
        if not list_container:
            print("No more pages.")
            break

        car_cards = list_container.select("a.box.vehicle-smallCard")
        if not car_cards:
            print("No cars found on page.")
            break

        for car in car_cards:
            if len(cars) >= TARGET_RECORDS:
                break

            try:
                h4 = car.find("h4")

                brand = h4.contents[0].strip()
                engine_type = h4.find("b").get_text(strip=True)

                is_new = "nový vůz" in h4.get_text().lower()

                params = car.select("ul.parameters li")

                transmission = None
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

                if is_new:
                    year = CURRENT_YEAR
                    km = 0

                price_raw = car.select_one(".prices b").get_text(strip=True)
                price = int(re.sub(r"[^\d]", "", price_raw))

                cars.append({
                    "brand": brand,
                    "engine_type": engine_type,
                    "fuel": fuel,
                    "transmission": transmission,
                    "km": km,
                    "price": price,
                    "year": year
                })

                # EVERY 20 RECORDS
                if len(cars) % 20 == 0:
                    save_progress()

                time.sleep(DELAY)

            except Exception as e:
                print("Skipped one car:", e)

        page += 1

except KeyboardInterrupt:
    print("\nInterrupted by user.")

except Exception as e:
    print("error:", e)

finally:
    save_progress()
    print("Scraping finished safely.")
