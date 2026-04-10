import os
import re
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# =============================
# CONFIG
# =============================
CSV_FILE = "../data/bazos_cars_labeled.csv"
IMAGE_DIR = "../data/car_images"
MODEL_DIR = "../models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 30
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 30

os.makedirs(MODEL_DIR, exist_ok=True)


# =============================
# LOAD DATA
# =============================
df = pd.read_csv(CSV_FILE, low_memory=False)
df["listing_id"] = df["listing_id"].astype(str)

# fallback image path
df["image_path"] = df["listing_id"].apply(
    lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg")
)

df = df.dropna(subset=["image_path", "brand", "model_extracted", "condition"])
df = df.reset_index(drop=True)

print("Rows:", len(df))



print("After removing rare brands:", len(df))

# ENCODE LABELS
brand_le = LabelEncoder()
model_le = LabelEncoder()
condition_le = LabelEncoder()

df["brand_enc"] = brand_le.fit_transform(df["brand"].astype(str))
df["model_enc"] = model_le.fit_transform(df["model_extracted"].astype(str))
df["condition_enc"] = condition_le.fit_transform(df["condition"].astype(str))

np.save(os.path.join(MODEL_DIR, "brand_classes"), brand_le.classes_)
np.save(os.path.join(MODEL_DIR, "model_classes"), model_le.classes_)
np.save(os.path.join(MODEL_DIR, "condition_classes"), condition_le.classes_)


brand_counts = df["brand_enc"].value_counts()

valid_brands = brand_counts[brand_counts >= 2].index
df = df[df["brand_enc"].isin(valid_brands)].reset_index(drop=True)

# =============================
# SPLIT
# =============================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["brand_enc"]
)


# =============================
# IMAGE LOADER
# =============================
def load_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img)
    return preprocess_input(img)


# =============================
# GENERATOR
# =============================
def data_generator(dataframe, batch_size=BATCH_SIZE, shuffle=True):
    while True:
        if shuffle:
            dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(dataframe), batch_size):
            batch = dataframe.iloc[i:i + batch_size]

            images, brand_y, model_y, cond_y = [], [], [], []

            for _, row in batch.iterrows():
                try:
                    images.append(load_image(row["image_path"]))
                    brand_y.append(row["brand_enc"])
                    model_y.append(row["model_enc"])
                    cond_y.append(row["condition_enc"])
                except:
                    continue

            if len(images) == 0:
                continue

            yield (
                np.array(images),
                {
                    "brand_output": np.array(brand_y),
                    "model_output": np.array(model_y),
                    "condition_output": np.array(cond_y),
                }
            )


# =============================
# MODEL
# =============================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# Stage 1 freeze
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

brand_output = Dense(len(brand_le.classes_), activation="softmax", name="brand_output")(x)
model_output = Dense(len(model_le.classes_), activation="softmax", name="model_output")(x)
condition_output = Dense(len(condition_le.classes_), activation="softmax", name="condition_output")(x)

model = Model(
    inputs=base_model.input,
    outputs=[brand_output, model_output, condition_output]
)


# =============================
# COMPILE STAGE 1
# =============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss={
        "brand_output": "sparse_categorical_crossentropy",
        "model_output": "sparse_categorical_crossentropy",
        "condition_output": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "brand_output": 0.3,
        "model_output": 1.0,
        "condition_output": 0.7,
    },
    metrics={
        "brand_output": ["accuracy"],
        "model_output": ["accuracy"],
        "condition_output": ["accuracy"],
    }
)

callbacks_stage1 = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, "stage1.keras"), save_best_only=True),
]

print("\n=== STAGE 1 ===")
model.fit(
    data_generator(train_df),
    validation_data=data_generator(val_df, shuffle=False),
    steps_per_epoch=max(1, len(train_df) // BATCH_SIZE),
    validation_steps=max(1, len(val_df) // BATCH_SIZE),
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks_stage1,
)


# =============================
# STAGE 2 FINETUNE
# =============================
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),
    loss={
        "brand_output": "sparse_categorical_crossentropy",
        "model_output": "sparse_categorical_crossentropy",
        "condition_output": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "brand_output": 0.3,
        "model_output": 1.0,
        "condition_output": 0.7,
    },
    metrics={
        "brand_output": ["accuracy"],
        "model_output": ["accuracy"],
        "condition_output": ["accuracy"],
    }
)

callbacks_stage2 = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, "final.keras"), save_best_only=True),
]

print("\n=== STAGE 2 ===")
model.fit(
    data_generator(train_df),
    validation_data=data_generator(val_df, shuffle=False),
    steps_per_epoch=max(1, len(train_df) // BATCH_SIZE),
    validation_steps=max(1, len(val_df) // BATCH_SIZE),
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks_stage2,
)


# =============================
# SAVE
# =============================
model.save(os.path.join(MODEL_DIR, "vision_model_final.keras"))
print("DONE ✅")