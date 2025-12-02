"""
Simple Perimeter Breach Detector (Side-Test version)
---------------------------------------------------
Logic:
 - User draws wall-top line on first frame.
 - YOLO11L + ByteTrack tracks people across frames.
 - For each tracked person:
     If centroid crosses from one side of the line to the other (sign change),
     the person is permanently labeled CROSSED (red box).
 - Others remain NORMAL (green box).

Requirements:
  pip install ultralytics opencv-python pandas
Run locally (GUI needed for OpenCV popup).
"""

import os, sys, math, json
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------- USER CONFIG ----------------
input_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/NewTestData"
output_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/Output-SideTest"
model_name = "yolo11l.pt"
conf_thresh = 0.30
resize_max_side = 960
# ------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

# ---- Load YOLO ----
print(f"Loading YOLO model: {model_name}")
model = YOLO(model_name)
print("✅ Model loaded with ByteTrack support.\n")

# ---- Mouse callback for line selection ----
CLICK_POINTS = []
def mouse_cb(event, x, y, flags, param):
    global CLICK_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINTS.append((int(x), int(y)))

def get_line_from_user(img_disp, window_title="Click two points on the wall top line"):
    """User clicks 2 points on the wall top line."""
    global CLICK_POINTS
    CLICK_POINTS = []
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(window_title, mouse_cb)
    display = img_disp.copy()
    while True:
        temp = display.copy()
        for p in CLICK_POINTS:
            cv2.circle(temp, p, 6, (0,255,255), -1)
        if len(CLICK_POINTS) >= 2:
            cv2.line(temp, CLICK_POINTS[0], CLICK_POINTS[1], (0,255,0), 2)
            msg = "Press 'c' confirm, 'r' reset, 's' skip, 'q' quit"
        else:
            msg = "Click 2 points on wall line. 's' skip, 'q' quit"
        cv2.putText(temp, msg, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(window_title, temp)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('r'):
            CLICK_POINTS = []
            continue
        if k == ord('s'):
            cv2.destroyWindow(window_title); return None
        if k == ord('q'):
            cv2.destroyWindow(window_title); return "QUIT"
        if k == ord('c') and len(CLICK_POINTS) >= 1:
            p1 = CLICK_POINTS[0]
            p2 = CLICK_POINTS[1] if len(CLICK_POINTS) > 1 else CLICK_POINTS[0]
            cv2.destroyWindow(window_title)
            return (p1, p2)

# ---- Helpers ----
def resize_for_display(img, max_side):
    if max_side is None:
        return img, 1.0
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h)), scale

def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def point_side(pt, p1, p2, eps=1e-9):
    """
    Returns:
      +1 if pt is on left side of line p1->p2,
      -1 if on right side,
       0 if approximately on the line.
    """
    x0, y0 = pt
    x1, y1 = p1
    x2, y2 = p2
    val = (x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1)
    if abs(val) <= eps:
        return 0
    return 1 if val > 0 else -1

# ---- Process videos ----
videos = [v for v in os.listdir(input_folder) if v.lower().endswith(".mp4")]
if not videos:
    print("No .mp4 files found in:", input_folder)
    sys.exit(0)

print(f"Found {len(videos)} videos to process.\n")

for idx, vid_name in enumerate(videos, 1):
    in_path = os.path.join(input_folder, vid_name)
    print(f"\n[{idx}/{len(videos)}] Processing {vid_name}")
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print("⚠️ Cannot open:", vid_name)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(output_folder, os.path.splitext(vid_name)[0] + "_annotated.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # ---- Draw line on first frame ----
    ret, first = cap.read()
    if not ret:
        print("⚠️ Cannot read first frame.")
        cap.release()
        writer.release()
        continue
    disp, scale = resize_for_display(first, resize_max_side)
    line_choice = get_line_from_user(disp, f"Draw wall top for {vid_name}")
    if line_choice == "QUIT":
        cap.release()
        writer.release()
        break
    if line_choice is None:
        cap.release()
        writer.release()
        continue
    p1_disp, p2_disp = line_choice

    # Save line JSON
    json_line_path = os.path.join(output_folder, os.path.splitext(vid_name)[0] + "_line.json")
    with open(json_line_path, "w") as jf:
        json.dump({"p1_disp": p1_disp, "p2_disp": p2_disp, "scale": scale}, jf)

    # ---- Tracking ----
    track_state = {}  # track_id -> {"crossed": bool, "last_side": int}
    rows = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if scale != 1.0:
            disp = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        else:
            disp = frame.copy()

        # YOLO with ByteTrack
        results = model.track(source=disp, persist=True, tracker="bytetrack.yaml",
                              conf=conf_thresh, verbose=False)
        if not results:
            break
        res = results[0]
        annotated = disp.copy()
        cv2.line(annotated, p1_disp, p2_disp, (255, 0, 0), 2)

        if hasattr(res, "boxes") and len(res.boxes) > 0:
            for b in res.boxes:
                if int(b.cls[0]) != 0:  # 0 = person
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                bbox = (x1, y1, x2, y2)
                cx, cy = centroid(bbox)
                side = point_side((cx, cy), p1_disp, p2_disp)
                track_id = int(b.id[0]) if b.id is not None else -1

                # initialize
                if track_id not in track_state:
                    track_state[track_id] = {"crossed": False, "last_side": side}

                # detect crossing (sign flip)
                last_side = track_state[track_id]["last_side"]
                if not track_state[track_id]["crossed"]:
                    if last_side != 0 and side != 0 and last_side != side:
                        track_state[track_id]["crossed"] = True
                if side != 0:
                    track_state[track_id]["last_side"] = side

                crossed = track_state[track_id]["crossed"]
                color = (0, 0, 255) if crossed else (0, 255, 0)
                label = "CROSSED" if crossed else "NORMAL"

                # draw
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(annotated, (int(cx), int(cy)), 4, (255, 255, 0), -1)
                cv2.putText(annotated, f"ID{track_id} {label}",
                            (int(x1), max(16, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                # record
                rows.append({
                    "video": vid_name, "frame": frame_id, "track_id": track_id,
                    "label": label, "side": side, "centroid_x": cx, "centroid_y": cy
                })

        annotated_full = cv2.resize(annotated, (w, h)) if scale != 1.0 else annotated
        writer.write(annotated_full)
        cv2.imshow("Processing...", cv2.resize(annotated_full, (min(960, w), min(540, h))))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Save CSV
    csv_path = os.path.join(output_folder, os.path.splitext(vid_name)[0] + "_detections.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"✅ Done: {vid_name}")
    print(f"   → Annotated video: {out_path}")
    print(f"   → Detections CSV:  {csv_path}")
    print(f"   → Line JSON:       {json_line_path}")

print("\nAll videos processed successfully.")
