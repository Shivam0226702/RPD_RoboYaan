import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

# ─── CONFIG ───────────────────────────────────────────
DATASET_PATH = "dataset"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
EPOCHS       = 30

# ─── DATA AUGMENTATION ────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.75, 1.25],
    shear_range=0.2,
    channel_shift_range=30.0,   
    fill_mode="nearest",
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Print class mapping
print("\n Class Mapping Detected:")
for cls, idx in train_data.class_indices.items():
    print(f"  {idx} → {cls}")

# CLASS WEIGHTS 
total = 1752 
weights = {
    0: total / (4 * 436),   
    1: total / (4 * 444),  
    2: total / (4 * 387),  
    3: total / (4 * 485),   
}
print(f"\n  Class weights: {weights}")

# MODEL 
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze 140 layers to reduce overfitting
for layer in base_model.layers[:140]:
    layer.trainable = False
for layer in base_model.layers[140:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# COMPILE 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003), 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# CALLBACKS 
callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "terrain_model_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=0.000001,
        verbose=1
    )
]

# TRAIN 
print("\n Training started...\n")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=weights,
    callbacks=callbacks
)

# SAVE 
model.save("terrain_model.h5")
print("\n Final model saved as terrain_model.h5")
print(" Best model saved as terrain_model_best.h5")

# CALCULATE MSE 
train_mse = [1 - acc for acc in history.history["accuracy"]]
val_mse   = [1 - acc for acc in history.history["val_accuracy"]]

# PLOTS
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

ax1.plot(history.history["accuracy"],     label="Train Accuracy", color="blue")
ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   color="orange")
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2.plot(history.history["loss"],     label="Train Loss", color="blue")
ax2.plot(history.history["val_loss"], label="Val Loss",   color="orange")
ax2.set_title("Model Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

ax3.plot(train_mse, label="Train MSE", color="blue")
ax3.plot(val_mse,   label="Val MSE",   color="orange")
ax3.set_title("Model MSE")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Mean Squared Error")
ax3.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()
print(" Training plot with MSE saved!")
