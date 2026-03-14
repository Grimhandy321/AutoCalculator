from concurrent.futures import ThreadPoolExecutor, as_completed

from aaautoScraper import AAAAutoScraper
from autoJarovScraper import AutoJarovScraper


def run_scraper(scraper):
    print(f"Starting {scraper.get_type()}")
    records = scraper.run()
    print(f"Finished {scraper.get_type()} ({len(records)} records)")
    return records


def run_all_scrapers():

    scrapers = [
        AutoJarovScraper(target_records=2000),
        AAAAutoScraper(target_records=2000),
    ]

    with ThreadPoolExecutor(max_workers=len(scrapers)) as executor:

        futures = [executor.submit(run_scraper, scraper) for scraper in scrapers]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("Scraper failed:", e)


if __name__ == "__main__":
    run_all_scrapers()