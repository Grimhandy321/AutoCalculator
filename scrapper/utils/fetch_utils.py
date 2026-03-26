import requests
import time
import random


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
}


def fetch_url(url, headers=None, timeout=10, retries=3, backoff=2, session=None):
    for attempt in range(1, retries + 1):
        try:
            if session:
                response = session.get(url, headers=headers, timeout=timeout)
            else:
                response = requests.get(url, headers=headers, timeout=timeout)

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            print(f"[fetch] attempt {attempt}/{retries} failed: {url} -> {e}")

            if attempt == retries:
                return None

            sleep_time = (backoff ** (attempt - 1)) + random.uniform(0.3, 1.0)
            time.sleep(sleep_time)

    return None