import tensorflow as tf
import numpy as np
import cv2
import os
import time

# ─── CONFIG ───────────────────────────────────────────
MODEL_PATH  = "terrain_model_best.h5"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["gravel", "rock_field", "sand", "smooth_ground"]

# ─── SPEED & RISK LOGIC ───────────────────────────────
def get_terrain_info(terrain, confidence):
    rules = {
        "smooth_ground": {
            "risk"       : "Safe",
            "speed_range": "70–100 km/h",
            "speed_min"  : 70,
            "speed_max"  : 100,
            "invert"     : False
        },
        "gravel": {
            "risk"       : "Moderate",
            "speed_range": "40–60 km/h",
            "speed_min"  : 40,
            "speed_max"  : 60,
            "invert"     : True
        },
        "sand": {
            "risk"       : "High",
            "speed_range": "20–40 km/h",
            "speed_min"  : 20,
            "speed_max"  : 40,
            "invert"     : True
        },
        "rock_field": {
            "risk"       : "Dangerous",
            "speed_range": "0–10 km/h",
            "speed_min"  : 0,
            "speed_max"  : 10,
            "invert"     : True
        }
    }

    info = rules[terrain]

    if confidence >= 85:
        factor = 1.0
    elif confidence >= 70:
        factor = 0.75
    elif confidence >= 55:
        factor = 0.5
    else:
        factor = 0.25

    if info["invert"]:
        factor = 1.0 - factor

    speed_min = info["speed_min"]
    speed_max = info["speed_max"]
    speed     = int(speed_min + (speed_max - speed_min) * factor)
    speed     = max(speed_min, min(speed_max, speed))

    return info["risk"], info["speed_range"], speed


# ─── SURFACE DESCRIPTION ──────────────────────────────
def get_terrain_description(terrain, confidence):
    descriptions = {
        "smooth_ground": {
            "high"  : "Surface is clean and flat — ideal for fast movement.",
            "medium": "Surface is mostly smooth but contains visible cracks or minor debris.",
            "low"   : "Surface appears smooth but has significant cracks or scattered gravel.",
            "vlow"  : "Terrain is mixed — smooth ground with heavy cracking or gravel coverage."
        },
        "gravel": {
            "high"  : "Dense gravel surface — consistent loose stones throughout.",
            "medium": "Gravel surface with some smoother patches mixed in.",
            "low"   : "Patchy gravel — mixed with sand or dirt sections.",
            "vlow"  : "Unclear gravel — surface blends with sand or rocky terrain."
        },
        "sand": {
            "high"  : "Pure sandy terrain — deep loose sand detected.",
            "medium": "Sandy surface with some firm or compacted patches.",
            "low"   : "Mixed sand — contains patches of gravel or hardened ground.",
            "vlow"  : "Uncertain sandy terrain — heavily mixed with other surface types."
        },
        "rock_field": {
            "high"  : "Dense rock field — large rocks and boulders throughout.",
            "medium": "Rocky terrain with some open gaps between rocks.",
            "low"   : "Scattered rocks — mixed with gravel or sandy patches.",
            "vlow"  : "Partially rocky — terrain blends between rock and gravel."
        }
    }

    if confidence >= 85:
        level = "high"
    elif confidence >= 70:
        level = "medium"
    elif confidence >= 55:
        level = "low"
    else:
        level = "vlow"

    return descriptions[terrain][level]


# ─── LOAD MODEL ───────────────────────────────────────
print("\n Loading terrain classification model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(" Model loaded successfully!")
print("=" * 50)


# ─── PREDICT SINGLE IMAGE ─────────────────────────────
def predict_terrain(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return

    img_resized  = cv2.resize(img, IMG_SIZE)
    img_array    = img_resized / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)

    start_time  = time.time()
    predictions = model.predict(img_expanded, verbose=0)
    end_time    = time.time()

    # ── Debug raw scores ─────────────────────────────
    print(f"\nRaw scores → gravel:{predictions[0][0]:.2f}  rock:{predictions[0][1]:.2f}  sand:{predictions[0][2]:.2f}  smooth:{predictions[0][3]:.2f}")

    # ── Temperature scaling (FIXED typo) ─────────────
    TEMPERATURE  = 1.3
    predictions  = predictions ** (1 / TEMPERATURE)   # ✅ fixed
    predictions  = predictions / np.sum(predictions)  # ✅ now correct

    inference_ms = (end_time - start_time) * 1000
    class_idx    = np.argmax(predictions[0])
    confidence   = float(np.max(predictions[0])) * 100
    terrain      = CLASS_NAMES[class_idx]

    risk, speed_range, speed = get_terrain_info(terrain, confidence)
    description              = get_terrain_description(terrain, confidence)
    display_name             = terrain.replace("_", " ").title()

    # ── Output ───────────────────────────────────────
    print(f"\n Image                : {os.path.basename(image_path)}")
    print(f"─" * 50)
    print(f"Terrain Detected        : {display_name}")
    print(f"Confidence              : {confidence:.0f}%")
    print(f"Surface Analysis        : {description}")
    print(f"─" * 50)
    print(f"Risk Level              : {risk}")
    print(f"Recommended Rover Speed : {speed} km/h")
    print(f"Speed Range             : {speed_range}")
    print(f"─" * 50)
    print(f"  Inference Time       : {inference_ms:.1f} ms")
    print(f"=" * 50)


# ─── PREDICT FOLDER ───────────────────────────────────
def predict_folder(folder_path):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images    = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(valid_ext)]

    if not images:
        print(" No images found in that folder!")
        return

    print(f"\n  Found {len(images)} image(s) in '{folder_path}'")
    print("=" * 50)

    for img_file in images:
        full_path = os.path.join(folder_path, img_file)
        predict_terrain(full_path)


# ─── MAIN ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Rover Terrain Classification System")
    print("=" * 50)
    print("  1 → Predict single image")
    print("  2 → Predict all images in a folder")
    print("=" * 50)
    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        path = input("Enter image path: ").strip().strip('"')
        predict_terrain(path)
    elif choice == "2":
        folder = input("Enter folder path: ").strip().strip('"')
        predict_folder(folder)
    else:
        print(" Invalid choice!")