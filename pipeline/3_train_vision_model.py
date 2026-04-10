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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from pipeline.utils.image_utils import find_augmented_image_paths, find_original_image_path

CSV_FILE = "../data/bazos_cars_labeled.csv"
IMAGE_DIR = "../data/car_images"
AUGMENT_DIR = "../data/car_images_augmented"
MODEL_DIR = "../models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 8
EPOCHS_STAGE2 = 8

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)
df["listing_id"] = df["listing_id"].astype(str)

# -----------------------------
# Load image paths
# -----------------------------
df["image_path"] = df["listing_id"].apply(find_original_image_path)

df = df.dropna(subset=["image_path", "brand", "model_extracted", "condition"]).reset_index(drop=True)

print("Usable original rows:", len(df))

# -----------------------------
# Encode labels
# -----------------------------
brand_le = LabelEncoder()
model_le = LabelEncoder()
condition_le = LabelEncoder()

df["brand_enc"] = brand_le.fit_transform(df["brand"].astype(str))
df["model_enc"] = model_le.fit_transform(df["model_extracted"].astype(str))
df["condition_enc"] = condition_le.fit_transform(df["condition"].astype(str))

# -----------------------------
# 🔥 REMOVE classes with < 2 samples (fix stratify)
# -----------------------------
counts = df["brand_enc"].value_counts()
valid_classes = counts[counts >= 2].index

df = df[df["brand_enc"].isin(valid_classes)].reset_index(drop=True)

print("Rows after removing rare brands:", len(df))

# -----------------------------
# Save encoders
# -----------------------------
np.save(os.path.join(MODEL_DIR, "brand_classes"), brand_le.classes_)
np.save(os.path.join(MODEL_DIR, "model_classes"), model_le.classes_)
np.save(os.path.join(MODEL_DIR, "condition_classes"), condition_le.classes_)

# -----------------------------
# Split BEFORE augmentation
# -----------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["brand_enc"]
)

# -----------------------------
# Expand train set with augmentations
# -----------------------------
def expand_with_augmentations(dataframe):
    rows = []

    for _, row in dataframe.iterrows():
        listing_id = row["listing_id"]

        # original
        rows.append({
            "image_path": row["image_path"],
            "brand_enc": row["brand_enc"],
            "model_enc": row["model_enc"],
            "condition_enc": row["condition_enc"]
        })

        # augmented
        aug_paths = find_augmented_image_paths(listing_id)
        for aug_path in aug_paths:
            rows.append({
                "image_path": aug_path,
                "brand_enc": row["brand_enc"],
                "model_enc": row["model_enc"],
                "condition_enc": row["condition_enc"]
            })

    return pd.DataFrame(rows)

train_expanded_df = expand_with_augmentations(train_df)

val_clean_df = val_df[["image_path", "brand_enc", "model_enc", "condition_enc"]].copy()

print("Training samples after augmentation:", len(train_expanded_df))
print("Validation samples:", len(val_clean_df))

# -----------------------------
# Image loading
# -----------------------------
def load_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

# -----------------------------
# Data generator
# -----------------------------
def data_generator(dataframe, batch_size=BATCH_SIZE, shuffle=True):
    while True:
        if shuffle:
            dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(dataframe), batch_size):
            batch = dataframe.iloc[i:i + batch_size]

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
                except Exception:
                    continue

            if len(images) == 0:
                continue

            yield (
                np.array(images, dtype=np.float32),
                {
                    "brand_output": np.array(brand_y, dtype=np.int32),
                    "model_output": np.array(model_y, dtype=np.int32),
                    "condition_output": np.array(condition_y, dtype=np.int32)
                }
            )

# -----------------------------
# Build model
# -----------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# Stage 1
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)

brand_output = Dense(len(brand_le.classes_), activation="softmax", name="brand_output")(x)
model_output = Dense(len(model_le.classes_), activation="softmax", name="model_output")(x)
condition_output = Dense(len(condition_le.classes_), activation="softmax", name="condition_output")(x)

model = Model(inputs=base_model.input, outputs=[brand_output, model_output, condition_output])

# -----------------------------
# Compile stage 1
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "brand_output": "sparse_categorical_crossentropy",
        "model_output": "sparse_categorical_crossentropy",
        "condition_output": "sparse_categorical_crossentropy"
    },
    metrics={
        "brand_output": ["accuracy"],
        "model_output": ["accuracy"],
        "condition_output": ["accuracy"]
    }
)

callbacks_stage1 = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, "vision_model_stage1.keras"), save_best_only=True)
]

print("\n=== STAGE 1 ===")
model.fit(
    data_generator(train_expanded_df),
    validation_data=data_generator(val_clean_df, shuffle=False),
    steps_per_epoch=max(1, len(train_expanded_df) // BATCH_SIZE),
    validation_steps=max(1, len(val_clean_df) // BATCH_SIZE),
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks_stage1
)

# -----------------------------
# Stage 2 (fine-tune)
# -----------------------------
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss={
        "brand_output": "sparse_categorical_crossentropy",
        "model_output": "sparse_categorical_crossentropy",
        "condition_output": "sparse_categorical_crossentropy"
    },
    metrics={
        "brand_output": ["accuracy"],
        "model_output": ["accuracy"],
        "condition_output": ["accuracy"]
    }
)

callbacks_stage2 = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, "vision_model.keras"), save_best_only=True)
]

print("\n=== STAGE 2 ===")
model.fit(
    data_generator(train_expanded_df),
    validation_data=data_generator(val_clean_df, shuffle=False),
    steps_per_epoch=max(1, len(train_expanded_df) // BATCH_SIZE),
    validation_steps=max(1, len(val_clean_df) // BATCH_SIZE),
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks_stage2
)

model.save(os.path.join(MODEL_DIR, "vision_model_final.keras"))
print("Saved vision model.")