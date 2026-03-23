import os
import requests
from tqdm import tqdm

def download_images(data, folder="data/images"):
    os.makedirs(folder, exist_ok=True)

    for i, car in enumerate(tqdm(data)):
        images = car.get("images", [])

        for j, url in enumerate(images[:3]):
            try:
                filename = f"{folder}/{i}_{j}.jpg"

                if os.path.exists(filename):
                    continue

                r = requests.get(url, timeout=5)
                with open(filename, "wb") as f:
                    f.write(r.content)

            except:
                continue
