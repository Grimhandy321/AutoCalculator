import pandas as pd
import os
import threading

OUTPUT_FILE = "../data/cars_dataset.csv"
lock = threading.Lock()

def save_progress(records):
    with lock:
        if os.path.exists(OUTPUT_FILE):
            existing = pd.read_csv(OUTPUT_FILE)
            df = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
        else:
            df = pd.DataFrame(records)

        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(df)} rows")