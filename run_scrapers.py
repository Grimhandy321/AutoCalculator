import json
import os

from scrapper.AAAAutoScraper import AAAAutoScraper
from scrapper.AutoJarovScraper import AutoJarovScraper


MAX_TOTAL = 2000
SAVE_PATH = "data/raw/cars.json"


def load_existing():
    if not os.path.exists(SAVE_PATH):
        return []
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run():
    scrapers = [
        AAAAutoScraper(),
        AutoJarovScraper()
    ]

    all_data = load_existing()

    total = len(all_data)
    print(f"Starting with {total} records")

    for scraper in scrapers:

        page = 1

        while total < MAX_TOTAL:
            print(f"[{scraper.get_type()}] page {page} | total={total}")

            try:
                page_data = scraper.scrape_page(page)

                if not page_data:
                    break

                for record in page_data:
                    if total >= MAX_TOTAL:
                        break

                    all_data.append(record)
                    total += 1

                    # save every 20 globally
                    if total % 20 == 0:
                        save_data(all_data)
                        print(f"Saved {total}")

                page += 1

            except Exception as e:
                print("Error:", e)
                break

    save_data(all_data)
    print(f"Done. Total records: {total}")


if __name__ == "__main__":
    run()