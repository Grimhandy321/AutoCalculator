from pipeline.dataset_cleaning import load_data, clean_data, save_clean
from pipeline.download_images import download_images

data = load_data("data/raw/cars.json")

df = clean_data(data)
save_clean(df)

download_images(data)
