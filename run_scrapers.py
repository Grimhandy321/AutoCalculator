import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from scrapper.AAAAutoScraper import AAAAutoScraper
from scrapper.AutoJarovScraper import AutoJarovScraper


MAX_TOTAL = 2000
SAVE_PATH = "data/raw/cars.json"
MAX_WORKERS = 2


def load_existing():
    if not os.path.exists(SAVE_PATH):
        return []
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scraper_worker(scraper, all_data, total_counter, data_lock, stop_event):
    page = 1

    while not stop_event.is_set():
        with data_lock:
            if total_counter["count"] >= MAX_TOTAL:
                stop_event.set()
                break
            current_total = total_counter["count"]

        print(f"[{scraper.get_type()}] page {page} | total={current_total}")

        try:
            page_data = scraper.scrape_page(page)

            if not page_data:
                print(f"[{scraper.get_type()}] no more data on page {page}")
                break

            with data_lock:
                for record in page_data:
                    if total_counter["count"] >= MAX_TOTAL:
                        stop_event.set()
                        break

                    all_data.append(record)
                    total_counter["count"] += 1

                    if total_counter["count"] % 20 == 0:
                        save_data(all_data)
                        print(f"Saved {total_counter['count']}")

            page += 1

        except Exception as e:
            print(f"[{scraper.get_type()}] Error on page {page}: {e}")
            break


def run():
    scrapers = [
       ## AAAAutoScraper(),
        AutoJarovScraper()
    ]

    all_data = load_existing()
    total_counter = {"count": len(all_data)}
    data_lock = threading.Lock()
    stop_event = threading.Event()

    print(f"Starting with {total_counter['count']} records")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(scraper_worker, scraper, all_data, total_counter, data_lock, stop_event)
            for scraper in scrapers
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("Worker failed:", e)

    save_data(all_data)
    print(f"Done. Total records: {total_counter['count']}")


if __name__ == "__main__":
    run()