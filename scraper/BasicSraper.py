from abc import ABC, abstractmethod
import time
from saver import save_progress


class BaseScraper(ABC):

    def __init__(self, target_records=2000, delay=0.1, save_interval=20):
        self.target_records = target_records
        self.delay = delay
        self.save_interval = save_interval
        self.records = []
        self.unsaved_records = []

    @abstractmethod
    def scrape_page(self, page):
        pass

    @abstractmethod
    def get_type(self):
        pass

    def run(self):

        page = 1

        try:
            while len(self.records) < self.target_records:

                print(f"{self.get_type()} -> page {page}")

                page_records = self.scrape_page(page)

                if not page_records:
                    break

                for r in page_records:

                    if len(self.records) >= self.target_records:
                        break

                    r["type"] = self.get_type()

                    self.records.append(r)
                    self.unsaved_records.append(r)

                    if len(self.unsaved_records) >= self.save_interval:
                        save_progress(self.unsaved_records)
                        self.unsaved_records = []

                page += 1
                time.sleep(self.delay)

        except KeyboardInterrupt:
            print(f"{self.get_type()} interrupted")

        finally:
            if self.unsaved_records:
                save_progress(self.unsaved_records)

        return self.records