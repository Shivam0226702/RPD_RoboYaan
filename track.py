import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from sort import Sort

YOLO_MODEL      = "yolov8n.pt"
CONF_THRESHOLD  = 0.40
IOU_THRESHOLD   = 0.45
MAX_AGE         = 10
MIN_HITS        = 2
BOX_COLOR       = (0, 220, 90)
TEXT_COLOR      = (255, 255, 255)
LABEL_BG_COLOR  = (0, 140, 55)
FONT            = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE      = 0.65
THICKNESS       = 2

target_lock   = threading.Lock()
target_object = ""
status_msg    = "Waiting for target"


def input_thread():
    global target_object, status_msg
    print("Roboyaan – Target Object Tracker")
    print("Type an object name and press Enter")
    print("Valid COCO names: person, bottle, laptop, book, chair, dog, cat")
    print("Press Q in the video window to quit\n")
    while True:
        raw = input("Enter target object: ").strip().lower()
        if raw:
            with target_lock:
                target_object = raw
                status_msg    = f"Searching for: {raw.upper()}"
            print(f"   -> Locking onto '{raw}' ...")


def draw_box(frame, x1, y1, x2, y2, label, conf, track_id):
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)

    corner_len = 15
    for cx, cy, dx, dy in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), BOX_COLOR, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), BOX_COLOR, 3)

    banner = f"{label.upper()}  {conf:.0%}  ID:{track_id}"
    (tw, th), _ = cv2.getTextSize(banner, FONT, FONT_SCALE, 1)
    by1 = max(y1 - th - 10, 0)
    cv2.rectangle(frame, (x1, by1), (x1 + tw + 8, y1), LABEL_BG_COLOR, -1)
    cv2.putText(frame, banner, (x1 + 4, y1 - 5), FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_hud(frame, fps, target, found_count):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS : {fps:.1f}", (10, 25), FONT, 0.58, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Target : {target.upper() if target else 'None'}",
                (10, 50), FONT, 0.58, (255, 220, 60), 1, cv2.LINE_AA)
    status = f"Tracking {found_count} object(s)" if found_count else "Searching..."
    color  = (80, 255, 80) if found_count else (80, 160, 255)
    cv2.putText(frame, f"Status : {status}", (10, 75), FONT, 0.58, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit", (10, 98), FONT, 0.45, (140, 140, 140), 1, cv2.LINE_AA)


def main():
    global target_object, status_msg

    print(f"\n[INFO] Loading {YOLO_MODEL} ...")
    model = YOLO(YOLO_MODEL)
    class_names = model.names
    print(f"[INFO] Model ready. {len(class_names)} classes available.")
    print(f"[INFO] Sample classes: {list(class_names.values())[:10]}\n")

    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check your camera index.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=0.3)
    fps_prev = time.time()
    fps      = 0.0

    print("[INFO] Camera open. Video window launching...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed. Retrying...")
            continue

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_prev, 1e-6))
        fps_prev = now

        with target_lock:
            current_target = target_object

        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]

        target_dets = []

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = class_names[cls_id].lower()
            if cls_name == current_target:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                target_dets.append([x1, y1, x2, y2, conf])

        dets_np = np.array(target_dets) if target_dets else np.empty((0, 5))

        tracks = tracker.update(dets_np)

        def best_conf(tx1, ty1, tx2, ty2):
            if not target_dets:
                return 0.0
            best, best_c = 1e9, 0.0
            for d in target_dets:
                dist = abs(d[0]-tx1) + abs(d[1]-ty1) + abs(d[2]-tx2) + abs(d[3]-ty2)
                if dist < best:
                    best, best_c = dist, d[4]
            return best_c

        found_count = len(tracks)
        for trk in tracks:
            x1, y1, x2, y2, tid = map(int, trk[:5])
            conf = best_conf(x1, y1, x2, y2)
            draw_box(frame, x1, y1, x2, y2, current_target or "?", conf, tid)
            print(
                f"  Target: {(current_target or '?').upper():<10} | "
                f"Conf: {conf:.0%}  | ID: {tid}  | Status: Tracking Active",
                end="\r"
            )

        if tracks.size == 0 and current_target:
            print(f"  Searching for '{current_target.upper()}' ...          ", end="\r")

        draw_hud(frame, fps, current_target, found_count)

        cv2.imshow("Roboyaan – Target Tracker  (Q to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n\n[INFO] Session ended. Goodbye!")


if __name__ == "__main__":
    main()
