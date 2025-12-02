"""
Multi-video perimeter detector (bbox-overlap-only)
- Uses YOLO11L + ByteTrack for tracking
- Draw ground polygon (multi-point, filled) on first frame (left-click add, right-click finish)
- For each person bbox compute bbox_overlap = intersection_area / bbox_area
- Label ON_GROUND if bbox_overlap >= BBOX_FRAC_THRESH else OFF_GROUND
- Draws bbox, centroid, polygon, saves annotated video + CSV + regions JSON

Requirements:
  pip install ultralytics opencv-python pandas shapely
Run locally (OpenCV GUI required).
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from ultralytics import YOLO

# ---------------- USER CONFIG ----------------
INPUT_FOLDER = "/Users/singhharshvardhan580/Developer/Perimeter Detection/singlevid"
OUTPUT_FOLDER = "/Users/singhharshvardhan580/Developer/Perimeter Detection/Output-GroundIoU-Fill-bisleri"
MODEL_NAME = "yolo11n.pt"
CONF_THRESH = 0.30
RESIZE_MAX_SIDE = 960        # set None to run at native resolution (slower)
BBOX_FRAC_THRESH = 0.01      # fraction of bbox area overlapping ground to call ON_GROUND (e.g., 0.30 = 30%)
FILL_COLOR_BGR = (150, 0, 0) # dark-blue BGR for fill
FILL_ALPHA = 0.45
OUTLINE_COLOR = (0, 255, 255)
# ------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading YOLO model:", MODEL_NAME)
model = YOLO(MODEL_NAME)
print("Model loaded.\n")

# ---------------- helpers: display resize ----------------
def resize_for_display(img, max_side):
    if max_side is None:
        return img, 1.0
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / float(max(h, w))
    return cv2.resize(img, (int(w * scale), int(h * scale))), scale

# ---------------- interactive polygon + optional line ----------------
def draw_filled_polygon(img_disp, window_title="Draw ground polygon: left-click add, right-click finish"):
    pts = []
    done = False
    cancelled = False

    def _mouse(event, x, y, flags, param):
        nonlocal pts, done
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(pts) >= 3:
                done = True

    win = window_title
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(win, _mouse)

    while True:
        temp = img_disp.copy()
        if len(pts) >= 2:
            cv2.polylines(temp, [np.array(pts, np.int32)], isClosed=False, color=OUTLINE_COLOR, thickness=2)
        for p in pts:
            cv2.circle(temp, p, 5, (0,255,255), -1)
        if len(pts) >= 3:
            poly_pts = np.array([pts], dtype=np.int32)
            overlay = temp.copy()
            cv2.fillPoly(overlay, poly_pts, FILL_COLOR_BGR)
            cv2.addWeighted(overlay, FILL_ALPHA, temp, 1 - FILL_ALPHA, 0, temp)
            cv2.polylines(temp, [poly_pts], isClosed=True, color=OUTLINE_COLOR, thickness=2)
        cv2.putText(temp, "Left-click add, Right-click finish, 'q' cancel", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(win, temp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            cancelled = True
            break
        if done:
            break

    cv2.destroyWindow(win)
    if cancelled or len(pts) < 3:
        return None
    return pts

def get_optional_line(img_disp, window_title="(Optional) Click two points for wall top, 's' skip"):
    CLICK_POINTS = []
    def mouse_cb(event, x, y, flags, param):
        nonlocal CLICK_POINTS
        if event == cv2.EVENT_LBUTTONDOWN:
            CLICK_POINTS.append((int(x), int(y)))
    win = window_title
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(win, mouse_cb)
    while True:
        temp = img_disp.copy()
        for p in CLICK_POINTS:
            cv2.circle(temp, p, 5, (0,255,255), -1)
        if len(CLICK_POINTS) >= 2:
            cv2.line(temp, CLICK_POINTS[0], CLICK_POINTS[1], (0,255,0), 2)
            msg = "Press 'c' confirm, 'r' reset, 's' skip, 'q' quit"
        else:
            msg = "Click two points or press 's' to skip. 'q' quit"
        cv2.putText(temp, msg, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(win, temp)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('r'):
            CLICK_POINTS = []
            continue
        if k == ord('s'):
            cv2.destroyWindow(win)
            return None
        if k == ord('q'):
            cv2.destroyWindow(win)
            return "QUIT"
        if k == ord('c') and len(CLICK_POINTS) >= 1:
            p1 = CLICK_POINTS[0]
            p2 = CLICK_POINTS[1] if len(CLICK_POINTS) > 1 else CLICK_POINTS[0]
            cv2.destroyWindow(win)
            return (p1, p2)

# ---------------- geometry: bbox-overlap-only ----------------
def compute_bbox_overlap_fraction(bbox, ground_poly):
    """
    Returns overlap fraction = intersection_area / bbox_area (0..1).
    """
    x1, y1, x2, y2 = bbox
    bbox_poly = box(x1, y1, x2, y2)
    if not bbox_poly.is_valid:
        return 0.0, 0.0, 0.0
    bbox_area = bbox_poly.area
    if bbox_area <= 0:
        return 0.0, 0.0, 0.0
    inter_area = ground_poly.intersection(bbox_poly).area if ground_poly.is_valid else 0.0
    frac = inter_area / bbox_area
    return frac, inter_area, bbox_area

# ---------------- main processing loop ----------------
videos = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".mp4")]
if not videos:
    print("No .mp4 files found in:", INPUT_FOLDER)
    sys.exit(0)
print(f"Found {len(videos)} videos to process.\n")

for vid_idx, vid_name in enumerate(videos, start=1):
    in_path = os.path.join(INPUT_FOLDER, vid_name)
    print(f"[{vid_idx}/{len(videos)}] {vid_name}")

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print("  ❌ Cannot open:", in_path)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(vid_name)[0] + "_annotated.mp4")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (orig_w, orig_h))

    # read first frame for interactive region drawing
    ret, first = cap.read()
    if not ret:
        print("  ❌ Cannot read first frame.")
        cap.release(); writer.release(); continue

    disp, scale = resize_for_display(first, RESIZE_MAX_SIDE)

    # optional wall line (kept for reference)
    line_choice = get_optional_line(disp, f"Optional: wall top for {vid_name}")
    if line_choice == "QUIT":
        cap.release(); writer.release(); break
    p1_disp, p2_disp = (line_choice or ((0,0), (0,0)))

    # draw ground polygon (multi-point, filled)
    ground_pts = draw_filled_polygon(disp, f"Draw ground polygon for {vid_name}")
    if ground_pts is None:
        print("  ⚠️ Ground polygon skipped or cancelled. Using empty polygon -> all OFF_GROUND.")
        ground_poly = Polygon()
    else:
        ground_poly = Polygon(ground_pts)

    # save regions JSON for reuse
    json_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(vid_name)[0] + "_regions.json")
    with open(json_path, "w") as jf:
        json.dump({"p1": p1_disp, "p2": p2_disp, "ground_pts": ground_pts, "scale": scale}, jf)

    rows = []
    frame_id = 0
    print(" Processing frames... (press 'q' in preview to stop early)")

    # run through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # resize to display scale (we run detection on 'disp')
        if scale != 1.0:
            disp_frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        else:
            disp_frame = frame.copy()

        # track (ByteTrack) via model.track
        results = model.track(source=disp_frame, persist=True, tracker="bytetrack.yaml",
                              conf=CONF_THRESH, verbose=False)
        if not results:
            break
        res = results[0]
        annotated = disp_frame.copy()

        # draw filled ground region if available
        if ground_pts:
            poly_pts = np.array([ground_pts], dtype=np.int32)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, poly_pts, FILL_COLOR_BGR)
            cv2.addWeighted(overlay, FILL_ALPHA, annotated, 1 - FILL_ALPHA, 0, annotated)
            cv2.polylines(annotated, [poly_pts], isClosed=True, color=OUTLINE_COLOR, thickness=2)

        # draw optional line
        if line_choice is not None:
            cv2.line(annotated, p1_disp, p2_disp, (255,0,0), 2)

        if hasattr(res, "boxes") and len(res.boxes) > 0:
            for b in res.boxes:
                # only persons (class 0)
                if int(b.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                bbox = (float(x1), float(y1), float(x2), float(y2))
                track_id = int(b.id[0]) if b.id is not None else -1
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)

                frac, inter_area, bbox_area = compute_bbox_overlap_fraction(bbox, ground_poly)
                label = "ON_GROUND" if frac >= BBOX_FRAC_THRESH else "OFF_GROUND"
                color = (0,255,0) if label == "ON_GROUND" else (0,0,255)

                # draw bbox, centroid
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(annotated, (cx, cy), 4, (255,255,0), -1)
                cv2.putText(annotated, f"ID{track_id} {label} {frac:.2f}",
                            (int(x1), max(16, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # map coordinates back to original if scaled
                if scale != 1.0:
                    x1o, y1o, x2o, y2o = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                    cxo, cyo = cx / scale, cy / scale
                else:
                    x1o, y1o, x2o, y2o, cxo, cyo = x1, y1, x2, y2, cx, cy

                rows.append({
                    "video": vid_name,
                    "frame": frame_id,
                    "track_id": track_id,
                    "label": label,
                    "bbox_overlap_frac": float(frac),
                    "inter_area": float(inter_area),
                    "bbox_area": float(bbox_area),
                    "x1": x1o, "y1": y1o, "x2": x2o, "y2": y2o,
                    "centroid_x": float(cxo), "centroid_y": float(cyo)
                })

        # upsample to original resolution and write
        if scale != 1.0:
            annotated_full = cv2.resize(annotated, (orig_w, orig_h))
        else:
            annotated_full = annotated
        writer.write(annotated_full)

        # preview
        preview = cv2.resize(annotated_full, (min(960, orig_w), min(540, orig_h)))
        cv2.imshow("Processing...", preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Early stop by user.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # save CSV + regions JSON already saved earlier
    csv_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(vid_name)[0] + "_detections.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: annotated video -> {out_video_path}")
    print(f"       CSV -> {csv_path}")
    print(f"       regions -> {json_path}")

print("All videos processed.")
