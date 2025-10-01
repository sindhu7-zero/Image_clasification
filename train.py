"""
Multiclass Fish Classifier - Fixed for Keras 3 / TensorFlow 2.16+ shape mismatch.
Optimized for 16GB RAM and Windows path handling.
"""

import os
# Set environment variables BEFORE importing TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_KERAS_IMAGE_DATA_FORMAT"] = "channels_last"

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure channels_last (HWC) format
tf.keras.backend.set_image_data_format("channels_last")

# Clear session to avoid stale graph issues
tf.keras.backend.clear_session()

# Configure GPU memory growth (if available)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("GPU config must happen before initialization:", e)

# Dataset path (use your exact folder name)
DATASET_ROOT = r"Image Clasification\images.cv_jzk6llhf18tm3k0kyttxz"
TRAIN_DIR = os.path.join(DATASET_ROOT, "data", "train")
VAL_DIR = os.path.join(DATASET_ROOT, "data", "val")

# Reduced batch size for 16GB RAM
BATCH_SIZE = 16

# Data generator
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_ds = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
)

val_ds = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

class_names = sorted(train_ds.class_indices.keys())
print("Classes:", class_names)

# Build model — now safe for Keras 3
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),  # RGB = 3 channels
)
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train
print("Starting training...")
model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)

# Save
os.makedirs("models", exist_ok=True)
model.save("models/fish_classifier.h5")
with open("models/class_names.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print("✅ Done!")