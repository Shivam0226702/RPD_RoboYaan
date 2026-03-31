# Roboyaan Competition – Task 2
## Target-Specific Object Detection and Tracking System

A real-time object detection and tracking system built for the Roboyaan Software Domain competition.  
The user types any object name, and the system finds **only that object** in the live camera feed, tracking it with a persistent ID.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10+ |
| Detection | YOLOv8 (Ultralytics) |
| Tracking | SORT (Kalman Filter + Hungarian Algorithm) |
| Video I/O | OpenCV |

---

## How It Works

```
Camera → YOLO detects all objects → Filter by user-typed class → SORT tracker → Draw bounding box
```

1. **Camera Input** – OpenCV reads the live feed frame by frame  
2. **Detect All** – YOLOv8 finds every object in the frame  
3. **Filter** – Only the user's target class passes through  
4. **Track** – SORT assigns a persistent ID and follows the object across frames  
5. **Display** – Bounding box with class name, confidence, and tracking ID is drawn  

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Shivam0226702/RPD_RoboYaan.git
cd RPD_RoboYaan

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the tracker
python track.py
```

When the window opens, type your target object name in the terminal (e.g. `bottle`, `person`, `laptop`) and press Enter.

---

## Controls

| Key | Action |
|-----|--------|
| Q or Esc | Quit the application |
| Type in terminal | Change target object |

---

## Valid Target Names (COCO classes)

```
person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird,
cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe,
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,
tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl,
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza,
donut, cake, chair, sofa, pottedplant, bed, diningtable, toilet,
tvmonitor, laptop, mouse, remote, keyboard, cell phone, microwave,
oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush
```

---

## Bonus Features Implemented

- ✅ **FPS display** – Real-time performance counter in HUD  
- ✅ **Tracking stability** – Kalman-filter smoothing via SORT  
- ✅ **Multi-target support** – Track all instances of the target class simultaneously  
- ✅ **Re-identification after loss** – `max_age=10` keeps tracks alive for 10 frames after occlusion  
- ✅ **Confidence display** – Each box shows detection confidence percentage  

---

## Output Format

```
Target: BOTTLE   | Conf: 92%  | ID: 3  | Status: Tracking Active
```

---

## File Structure

```
.
├── track.py          # Main detection + tracking script
├── sort.py           # SORT tracker (Kalman Filter + Hungarian matching)
├── requirements.txt  # Python dependencies
└── README.md
```
