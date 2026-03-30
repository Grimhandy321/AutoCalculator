import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

CSV_FILE = "../data/bazos_cars_labeled.csv"
IMAGE_DIR = "../data/car_images"
MODEL_DIR = "../models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)
df["listing_id"] = df["listing_id"].astype(str)

def find_image_path(listing_id):
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = os.path.join(IMAGE_DIR, f"{listing_id}{ext}")
        if os.path.exists(path):
            return path
    return None

df["image_path"] = df["listing_id"].apply(find_image_path)
df = df.dropna(subset=["image_path", "brand", "model", "condition"]).reset_index(drop=True)

print("Usable rows:", len(df))

# Encode labels
brand_le = LabelEncoder()
model_le = LabelEncoder()
condition_le = LabelEncoder()

df["brand_enc"] = brand_le.fit_transform(df["brand"])
df["model_enc"] = model_le.fit_transform(df["model"])
df["condition_enc"] = condition_le.fit_transform(df["condition"])

np.save(os.path.join(MODEL_DIR, "brand_classes.npy"), brand_le.classes_)
np.save(os.path.join(MODEL_DIR, "model_classes.npy"), model_le.classes_)
np.save(os.path.join(MODEL_DIR, "condition_classes.npy"), condition_le.classes_)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

def load_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

def data_generator(dataframe, batch_size=BATCH_SIZE, shuffle=True):
    while True:
        if shuffle:
            dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(dataframe), batch_size):
            batch = dataframe.iloc[i:i+batch_size]

            images = []
            brand_y = []
            model_y = []
            condition_y = []

            for _, row in batch.iterrows():
                try:
                    img = load_image(row["image_path"])
                    images.append(img)
                    brand_y.append(row["brand_enc"])
                    model_y.append(row["model_enc"])
                    condition_y.append(row["condition_enc"])
                except:
                    continue

            if len(images) == 0:
                continue

            yield (
                np.array(images),
                {
                    "brand_output": np.array(brand_y),
                    "model_output": np.array(model_y),
                    "condition_output": np.array(condition_y)
                }
            )

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)

brand_output = Dense(len(brand_le.classes_), activation="softmax", name="brand_output")(x)
model_output = Dense(len(model_le.classes_), activation="softmax", name="model_output")(x)
condition_output = Dense(len(condition_le.classes_), activation="softmax", name="condition_output")(x)

model = Model(inputs=base_model.input, outputs=[brand_output, model_output, condition_output])

model.compile(
    optimizer="adam",
    loss={
        "brand_output": "sparse_categorical_crossentropy",
        "model_output": "sparse_categorical_crossentropy",
        "condition_output": "sparse_categorical_crossentropy"
    },
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ModelCheckpoint(os.path.join(MODEL_DIR, "vision_model.keras"), save_best_only=True)
]

model.fit(
    data_generator(train_df),
    validation_data=data_generator(val_df, shuffle=False),
    steps_per_epoch=max(1, len(train_df)//BATCH_SIZE),
    validation_steps=max(1, len(val_df)//BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

model.save(os.path.join(MODEL_DIR, "vision_model_final.keras"))
print("Saved vision model.")