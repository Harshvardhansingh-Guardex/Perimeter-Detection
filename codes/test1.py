import os, sys, time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ------------------- USER CONFIG -------------------
input_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/Test Data"     # <-- set input folder
output_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/Output-test1"   # <-- set output folder
model_name = "yolo11l.pt"                  # change model if desired
conf_thresh = 0.30
resize_max_side = 1280    # max side to resize for display/inference (maintain aspect); set None to skip
# -------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

# load model
print("Loading YOLO model:", model_name)
model = YOLO(model_name)
print("Model loaded.")

# helper: resize for display while preserving scale
def resize_for_display(img, max_side):
    if max_side is None:
        return img, 1.0
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_res = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img_res, scale
    return img, 1.0

# mouse callback to capture two clicks
CLICK_POINTS = []
def mouse_callback(event, x, y, flags, param):
    global CLICK_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINTS.append((int(x), int(y)))

def get_wall_line_popup(img_bgr, window_name="Define wall top - click left then right"):
    """
    Shows image in a popup, user clicks two points along wall top.
    Returns wall_top_y (in image coords), or None if skipped or quit.
    """
    global CLICK_POINTS
    CLICK_POINTS = []
    display_img = img_bgr.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = display_img.copy()
        # draw clicked points
        for p in CLICK_POINTS:
            cv2.circle(temp, p, 6, (0,255,255), -1)
        # if two points, draw line
        if len(CLICK_POINTS) >= 2:
            cv2.line(temp, CLICK_POINTS[0], CLICK_POINTS[1], (0,255,0), 2)
            # show text: press 'c' to confirm
            cv2.putText(temp, "Press 'c' to confirm, 'r' to reset, 's' skip, 'q' quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.putText(temp, "Click two points along wall top (left then right). 's' skip, 'q' quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow(window_name, temp)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('r'):
            CLICK_POINTS = []
            display_img = img_bgr.copy()
            continue
        if key == ord('s'):
            cv2.destroyWindow(window_name)
            return None  # skip this image
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return "QUIT"
        if key == ord('c'):
            if len(CLICK_POINTS) >= 1:
                # Use average y of clicked points as wall_top_y (if only one clicked, use that y)
                ys = [p[1] for p in CLICK_POINTS]
                wall_top_y = int(round(np.mean(ys)))
                cv2.destroyWindow(window_name)
                return wall_top_y
            else:
                # ignore confirm if no points yet
                continue

# classification by normalized centroid distance
def classify_by_norm_centroid(bbox, wall_top_y):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h = max(1.0, (y2 - y1))
    norm_distance = (cy - float(wall_top_y)) / h
    if norm_distance < 0.0:
        label = "CROSSED"
    elif norm_distance < 0.2:
        label = "CLIMBING"
    else:
        label = "NORMAL"
    return label, float(norm_distance), float(cx), float(cy)

# iterate images
img_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
if len(img_files) == 0:
    print("No images found in input folder:", input_folder)
    sys.exit(0)

rows = []
print(f"Found {len(img_files)} images. For each image: click 2 points on wall top, press 'c' to confirm.")

for idx, name in enumerate(img_files, start=1):
    in_path = os.path.join(input_folder, name)
    print(f"\n[{idx}/{len(img_files)}] Processing: {name}")
    img_bgr_orig = cv2.imread(in_path)
    if img_bgr_orig is None:
        print("  ❌ Cannot read image, skipping.")
        continue

    # resize for display & inference
    img_disp, scale = resize_for_display(img_bgr_orig, resize_max_side)
    # get wall top via popup (on resized image)
    wall_y_disp = get_wall_line_popup(img_disp, window_name=f"Define wall top: {name}")
    if wall_y_disp == "QUIT":
        print("User requested quit. Exiting.")
        break
    if wall_y_disp is None:
        print("  Skipped by user.")
        continue

    # convert wall_y back to original image coordinates
    if scale != 1.0:
        wall_y_orig = int(round(wall_y_disp / scale))
    else:
        wall_y_orig = int(wall_y_disp)
    print(f"  wall_top_y (orig coords) = {wall_y_orig}")

    # Run YOLO detection on original-sized image (or you can run on resized)
    # We'll run on the resized image for speed and map boxes back to orig if scale !=1
    res = model(img_disp, conf=conf_thresh)[0]
    annotated = img_disp.copy()
    # draw the selected wall line on display image
    cv2.line(annotated, (0, wall_y_disp), (annotated.shape[1]-1, wall_y_disp), (255,0,0), 2)
    cv2.putText(annotated, f"WALL_Y={wall_y_disp} (disp coords)", (10, max(20, wall_y_disp-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    det_count = 0
    if hasattr(res, "boxes") and len(res.boxes) > 0:
        for b in res.boxes:
            # retrieve class and confidence
            cls_id = int(b.cls[0]) if hasattr(b.cls, 'cpu') else int(b.cls)
            conf = float(b.conf[0]) if hasattr(b.conf, 'cpu') else float(b.conf)
            names = res.names if hasattr(res, 'names') else None
            cls_name = names.get(cls_id, str(cls_id)) if names is not None else str(cls_id)
            if cls_name.lower() != "person":
                continue
            # get bbox in display coords
            xyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy, 'cpu') else np.array(b.xyxy)
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            # classify (on display coords)
            label, norm_d, cx, cy = classify_by_norm_centroid((x1,y1,x2,y2), wall_y_disp)
            det_count += 1
            # annotate
            color = (0,0,255) if label != "NORMAL" else (0,255,0)
            cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            txt = f"{label} {norm_d:.2f} conf:{conf:.2f}"
            cv2.putText(annotated, txt, (int(x1), max(16, int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # add CSV row: map boxes back to original coords if needed
            if scale != 1.0:
                x1o = int(round(x1 / scale)); y1o = int(round(y1 / scale))
                x2o = int(round(x2 / scale)); y2o = int(round(y2 / scale))
                cxo = cx / scale; cyo = cy / scale
                wall_y_for_csv = wall_y_orig
            else:
                x1o, y1o, x2o, y2o = int(x1), int(y1), int(x2), int(y2)
                cxo, cyo = cx, cy
                wall_y_for_csv = wall_y_disp

            rows.append({
                "image": name,
                "x1": x1o, "y1": y1o, "x2": x2o, "y2": y2o,
                "conf": float(conf),
                "label": label,
                "norm_distance": float(norm_d),
                "centroid_x": float(cxo),
                "centroid_y": float(cyo),
                "wall_top_y": int(wall_y_for_csv)
            })
    else:
        print("  No detections found.")

    # save annotated image to output folder (save resized annotated to keep what user saw)
    out_path = os.path.join(output_folder, name)
    cv2.imwrite(out_path, annotated)
    print(f"  Done. Detections={det_count}. Saved annotated image to: {out_path}")

# save CSV
csv_out = os.path.join(output_folder, "detections_summary.csv")
df = pd.DataFrame(rows)
df.to_csv(csv_out, index=False)
print("\nAll finished. CSV saved to:", csv_out)
cv2.destroyAllWindows()