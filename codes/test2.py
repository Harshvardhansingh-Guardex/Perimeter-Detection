"""
Interactive exact-line classifier (per-image).
- Click TWO points exactly on the line you want to use as reference.
- Classification uses signed perpendicular distance from bbox centroid to that line.
Requirements:
  pip install ultralytics opencv-python pandas --quiet
Run locally (not headless Colab).
"""

import os, sys
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import math

# ---------------- USER CONFIG ----------------
input_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/Test Data"     # <-- set input folder
output_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/Output-test2"
model_name = "yolo11l.pt"                  # or change to a smaller model if desired
conf_thresh = 0.35
resize_max_side = 1280   # for display; line clicks are on resized image and mapped back
climbing_margin_px = 20  # distance threshold (px) from exact line to label "CLIMBING"
# ------------------------------------------------

os.makedirs(output_folder, exist_ok=True)
model = YOLO(model_name)
print("Model loaded:", model_name)

# ---------- mouse callback to collect click points ----------
CLICK_POINTS = []
def mouse_cb(event, x, y, flags, param):
    global CLICK_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINTS.append((int(x), int(y)))

def resize_for_display(img, max_side):
    if max_side is None:
        return img, 1.0
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / float(max(h, w))
    new_w = int(round(w * scale)); new_h = int(round(h * scale))
    img_res = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_res, scale

def get_line_from_user(img_disp, window_title="Click two points on the exact line (left then right)"):
    """
    Shows img_disp, user clicks two points. Press:
      'c' to confirm (needs >=1 click; if one click used, its Y used but still exact is better with two),
      'r' to reset, 's' to skip image, 'q' to quit.
    Returns:
      None -> skipped
      "QUIT" -> quit
      (p1_disp, p2_disp) -> two points in display image coords (if only one clicked, second = same)
    """
    global CLICK_POINTS
    CLICK_POINTS = []
    win = window_title
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(win, mouse_cb)
    display = img_disp.copy()
    while True:
        tmp = display.copy()
        # draw click points
        for p in CLICK_POINTS:
            cv2.circle(tmp, p, 6, (0,255,255), -1)
        if len(CLICK_POINTS) >= 2:
            cv2.line(tmp, CLICK_POINTS[0], CLICK_POINTS[1], (0,255,0), 2)
            cv2.putText(tmp, "Press 'c' to confirm, 'r' to reset, 's' skip, 'q' quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.putText(tmp, "Click two points on the exact line. 's' skip, 'q' quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(win, tmp)
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
        if k == ord('c'):
            if len(CLICK_POINTS) == 0:
                continue
            if len(CLICK_POINTS) == 1:
                p1 = CLICK_POINTS[0]
                p2 = CLICK_POINTS[0]
            else:
                p1, p2 = CLICK_POINTS[0], CLICK_POINTS[1]
            cv2.destroyWindow(win)
            return (p1, p2)

# ---------- geometry helpers ----------
def centroid_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def signed_distance_point_to_line(pt, p1, p2):
    # line given by p1->p2, return signed perp distance (pixels)
    x0, y0 = pt
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    denom = math.hypot(a, b)
    if denom == 0:
        return 0.0
    d = (a * x0 + b * y0 + c) / denom
    return d

def classify_by_exact_line(bbox, p1_disp, p2_disp, margin_px):
    ctr = centroid_from_bbox(bbox)
    d = signed_distance_point_to_line(ctr, p1_disp, p2_disp)
    # d sign depends on p1->p2 orientation; we treat negative as one side, positive other
    if d < 0:
        label = "CROSSED"
    elif d <= margin_px:
        label = "CLIMBING"
    else:
        label = "NORMAL"
    return label, d

# ---------- main interactive loop ----------
img_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
if not img_files:
    print("No images found in:", input_folder); sys.exit(0)

rows = []
print("Interactive exact-line classifier")
print("For each image: click two points exactly on the line, then press 'c' to confirm.")

for idx, fname in enumerate(img_files, start=1):
    in_path = os.path.join(input_folder, fname)
    img_bgr = cv2.imread(in_path)
    if img_bgr is None:
        print(f"Skipping unreadable: {fname}"); continue

    img_disp, scale = resize_for_display(img_bgr, resize_max_side)
    user_choice = get_line_from_user(img_disp, window_title=f"Define exact line - {fname}")
    if user_choice == "QUIT":
        print("User quit."); break
    if user_choice is None:
        print(f"Skipped by user: {fname}"); continue

    p1_disp, p2_disp = user_choice
    # Map display points back to original image coordinates
    if scale != 1.0:
        p1_orig = (p1_disp[0] / scale, p1_disp[1] / scale)
        p2_orig = (p2_disp[0] / scale, p2_disp[1] / scale)
    else:
        p1_orig = p1_disp
        p2_orig = p2_disp

    # Run detection on display image for speed (boxes in display coords)
    res = model(img_disp, conf=conf_thresh)[0]

    annotated = img_disp.copy()
    # draw exact line in display coords
    cv2.line(annotated, p1_disp, p2_disp, (255,0,0), 2)
    cv2.putText(annotated, f"Exact line (display coords)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    det_count = 0
    if hasattr(res, "boxes") and len(res.boxes) > 0:
        for b in res.boxes:
            # class and conf
            cls_id = int(b.cls[0]) if hasattr(b.cls, 'cpu') else int(b.cls)
            conf = float(b.conf[0]) if hasattr(b.conf, 'cpu') else float(b.conf)
            names = res.names if hasattr(res, 'names') else None
            cls_name = names.get(cls_id, str(cls_id)) if names is not None else str(cls_id)
            if cls_name.lower() != "person":
                continue
            # bbox in display coords
            xyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy, 'cpu') else np.array(b.xyxy)
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            bbox_disp = (x1, y1, x2, y2)
            label, dist = classify_by_exact_line(bbox_disp, p1_disp, p2_disp, climbing_margin_px)
            det_count += 1
            # draw bbox, centroid and distance
            cx = int((x1+x2)/2); cy = int((y1+y2)/2)
            color = (0,255,0) if label == "NORMAL" else ((0,165,255) if label=="CLIMBING" else (0,0,255))
            cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            cv2.circle(annotated, (cx, cy), 4, (255,255,0), -1)
            txt = f"{label} d={dist:.1f} conf:{conf:.2f}"
            cv2.putText(annotated, txt, (int(x1), max(16, int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Map bbox & centroid back to original coords for CSV
            if scale != 1.0:
                x1o = int(round(x1/scale)); y1o = int(round(y1/scale))
                x2o = int(round(x2/scale)); y2o = int(round(y2/scale))
                cxo = (cx/scale); cyo = (cy/scale)
                # distance in display px -> convert to original approx by /scale
                dist_orig = dist / scale
            else:
                x1o, y1o, x2o, y2o = int(x1), int(y1), int(x2), int(y2)
                cxo, cyo = cx, cy
                dist_orig = dist

            rows.append({
                "image": fname,
                "x1": x1o, "y1": y1o, "x2": x2o, "y2": y2o,
                "conf": float(conf),
                "label": label,
                "distance_px_display": float(dist),
                "distance_px_orig": float(dist_orig),
                "centroid_x_disp": float(cx), "centroid_y_disp": float(cy),
                "centroid_x_orig": float(cxo), "centroid_y_orig": float(cyo),
                "p1_disp_x": p1_disp[0], "p1_disp_y": p1_disp[1],
                "p2_disp_x": p2_disp[0], "p2_disp_y": p2_disp[1]
            })
    else:
        print("  No person detections.")

    out_path = os.path.join(output_folder, fname)
    cv2.imwrite(out_path, annotated)
    print(f"[{idx}/{len(img_files)}] {fname}: detections={det_count} saved -> {out_path}")

# save CSV
csv_out = os.path.join(output_folder, "detections_exactline_summary.csv")
df = pd.DataFrame(rows)
df.to_csv(csv_out, index=False)
print("CSV written to:", csv_out)
cv2.destroyAllWindows()
