"""
Process all .mp4 videos in an input folder.
For each video:
 - Ask user to draw an exact line on the first frame.
 - Run YOLO11L person detection.
 - Classify using perpendicular distance to the clicked line.
 - Save annotated video, CSV, and clicked line JSON.

Requirements:
  pip install ultralytics opencv-python pandas
Run locally (GUI needed for OpenCV popup).
"""

import os, sys, math, time, json
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------- USER CONFIG ----------------
input_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/NewTestData"
output_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/Output-Multi"
model_name = "yolo11l.pt"
conf_thresh = 0.30
resize_max_side = 960       # resize for faster inference (set None for full-res)
climbing_margin_px = 20     # distance (px) threshold for CLIMBING
# ------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

# ---- Load YOLO model ----
print(f"Loading YOLO model: {model_name}")
model = YOLO(model_name)
print("✅ Model loaded.")

# ---- Mouse click logic ----
CLICK_POINTS = []
def mouse_cb(event, x, y, flags, param):
    global CLICK_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINTS.append((int(x), int(y)))

def get_line_from_user(img_disp, window_title="Click two points on the exact line (left then right)"):
    """Let user click two points for the reference line."""
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
            msg = "Click 2 points for the line. 's' skip, 'q' quit"
        cv2.putText(temp, msg, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(window_title, temp)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('r'):
            CLICK_POINTS = []
            continue
        if k == ord('s'):
            cv2.destroyWindow(window_title)
            return None
        if k == ord('q'):
            cv2.destroyWindow(window_title)
            return "QUIT"
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
    new_w, new_h = int(round(w*scale)), int(round(h*scale))
    return cv2.resize(img, (new_w, new_h)), scale

def centroid(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)/2, (y1+y2)/2)

def signed_distance(pt, p1, p2):
    x0,y0 = pt; x1,y1 = p1; x2,y2 = p2
    a, b, c = (y1-y2), (x2-x1), (x1*y2 - x2*y1)
    denom = math.hypot(a,b)
    if denom == 0: return 0
    return (a*x0 + b*y0 + c) / denom

def classify_by_line(bbox, p1, p2, margin):
    d = signed_distance(centroid(bbox), p1, p2)
    if d < 0:
        return "CROSSED", d
    elif d <= margin:
        return "CLIMBING", d
    return "NORMAL", d

# ---- Process all videos ----
videos = [v for v in os.listdir(input_folder) if v.lower().endswith(".mp4")]
if not videos:
    print("No .mp4 files found in:", input_folder); sys.exit(0)
print(f"Found {len(videos)} videos to process.\n")

for vid_idx, vid_name in enumerate(videos, start=1):
    in_path = os.path.join(input_folder, vid_name)
    print(f"\n[{vid_idx}/{len(videos)}] Processing {vid_name}")

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print("⚠️ Cannot open:", in_path); continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(output_folder, os.path.splitext(vid_name)[0] + "_annotated.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

    # ---- Get line on first frame ----
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not read first frame."); cap.release(); writer.release(); continue
    disp, scale = resize_for_display(frame, resize_max_side)
    line_choice = get_line_from_user(disp, window_title=f"Draw line for {vid_name}")
    if line_choice == "QUIT":
        print("User quit."); cap.release(); writer.release(); break
    if line_choice is None:
        print("Skipped video:", vid_name); cap.release(); writer.release(); continue
    p1_disp, p2_disp = line_choice
    p1_orig = (p1_disp[0]/scale, p1_disp[1]/scale)
    p2_orig = (p2_disp[0]/scale, p2_disp[1]/scale)

    # Save clicked line JSON
    json_line_path = os.path.join(output_folder, os.path.splitext(vid_name)[0] + "_line.json")
    with open(json_line_path, "w") as jf:
        json.dump({"p1_disp": p1_disp, "p2_disp": p2_disp, "p1_orig": p1_orig, "p2_orig": p2_orig, "scale": scale}, jf)

    # ---- Process all frames ----
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rows = []
    frame_id = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1

        if scale != 1.0:
            disp = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
        else:
            disp = frame.copy()

        # inference
        res = model(disp, conf=conf_thresh, verbose=False)[0]
        annotated = disp.copy()
        cv2.line(annotated, p1_disp, p2_disp, (255,0,0), 2)

        det_count = 0
        if hasattr(res, "boxes") and len(res.boxes) > 0:
            for b in res.boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                names = res.names
                label_name = names.get(cls_id, str(cls_id))
                if label_name.lower() != "person": continue
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                bbox = (x1,y1,x2,y2)
                label, dist = classify_by_line(bbox, p1_disp, p2_disp, climbing_margin_px)
                color = (0,255,0) if label=="NORMAL" else ((0,165,255) if label=="CLIMBING" else (0,0,255))
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                cv2.circle(annotated, (cx,cy), 4, (255,255,0), -1)
                txt = f"{label} d={dist:.1f}"
                cv2.putText(annotated, txt, (int(x1), max(16,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                det_count += 1

                # record for CSV
                if scale != 1.0:
                    x1o,y1o,x2o,y2o = x1/scale,y1/scale,x2/scale,y2/scale
                    cxo,cyo = cx/scale, cy/scale
                    dist_orig = dist/scale
                else:
                    x1o,y1o,x2o,y2o,cxo,cyo,dist_orig = x1,y1,x2,y2,cx,cy,dist
                rows.append({
                    "video": vid_name, "frame": frame_id,
                    "x1": x1o, "y1": y1o, "x2": x2o, "y2": y2o,
                    "conf": conf, "label": label,
                    "dist_display": dist, "dist_orig": dist_orig,
                    "centroid_x": cxo, "centroid_y": cyo
                })

        # upscale annotated frame back to original
        if scale != 1.0:
            annotated_full = cv2.resize(annotated, (orig_w, orig_h))
        else:
            annotated_full = annotated
        writer.write(annotated_full)

        # show quick preview
        cv2.imshow("Processing...", cv2.resize(annotated_full, (min(960,orig_w), min(540,orig_h))))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped early by user.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # save CSV
    csv_path = os.path.join(output_folder, os.path.splitext(vid_name)[0] + "_detections.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"✅ Done: {vid_name}")
    print(f"   → Annotated video: {out_path}")
    print(f"   → Detections CSV:  {csv_path}")
    print(f"   → Clicked line:    {json_line_path}")

print("\nAll videos processed successfully.")
