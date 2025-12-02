"""
Multi-video exact-line classifier with simple FP reduction and track confirmation.

- Click two points on the first frame of each video (exact line reference).
- Per-frame: run YOLO, compute signed perpendicular distance from bbox centroid to line.
- Zones:
    Outside (d < 0):
        if abs(d) > NEAR_BOUNDARY_THRESHOLD -> NEAR BOUNDARY (yellow)
        elif abs(d) < CLIMBING_THRESHOLD -> CLIMBING (red) candidate
        else -> OUTSIDE (green)
    Inside (d >= 0):
        if abs(d) < CLIMBING_THRESHOLD -> CLIMBING (orange)
        else -> INSIDE (green)
- Temporal confirmation: require sustained negative distance (d < 0) for CONFIRM_FRAMES (per-track) to raise intrusion alarm.
- Crowd filter: if many people in near zone, raise required confirmation.
"""

import os, sys, math, time, json
from collections import deque
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------- USER CONFIG ----------------
input_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/bisleri-video"
output_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/Output-Multi"
model_name = "yolo11n.pt"
conf_thresh = 0.30
resize_max_side = 960       # resize for display & inference
CLIMBING_THRESHOLD = 20     # px: < this => CLIMBING (near the line)
NEAR_BOUNDARY_THRESHOLD = 60 # px: outside & abs(d)>this => NEAR BOUNDARY
MIN_BBOX_AREA_RATIO = 0.0005 # ignore boxes smaller than this * image_area
MIN_CONF = 0.25             # further per-detection filter
ASPECT_RATIO_MIN = 0.35     # h/w
ASPECT_RATIO_MAX = 3.0
SUSTAIN_SECONDS = 0.6       # seconds to confirm intrusion (per-track)
CROWD_MAX = 3               # if people in near zone > CROWD_MAX, require stricter confirmation
# ------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

print("Loading YOLO model:", model_name)
model = YOLO(model_name)
print("Model loaded.")

# ---------------- mouse click logic ----------------
CLICK_POINTS = []
def mouse_cb(event, x, y, flags, param):
    global CLICK_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINTS.append((int(x), int(y)))

def get_line_from_user(img_disp, window_title="Click two points on the exact line (left then right)"):
    global CLICK_POINTS
    CLICK_POINTS = []
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(window_title, mouse_cb)
    display = img_disp.copy()
    while True:
        tmp = display.copy()
        for p in CLICK_POINTS:
            cv2.circle(tmp, p, 6, (0,255,255), -1)
        if len(CLICK_POINTS) >= 2:
            cv2.line(tmp, CLICK_POINTS[0], CLICK_POINTS[1], (0,255,0), 2)
            msg = "Press 'c' confirm, 'r' reset, 's' skip, 'q' quit"
        else:
            msg = "Click 2 points for the line. 's' skip, 'q' quit"
        cv2.putText(tmp, msg, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(window_title, tmp)
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

# ---------------- geometry helpers ----------------
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
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def signed_distance(pt, p1, p2):
    x0,y0 = pt; x1,y1 = p1; x2,y2 = p2
    a, b, c = (y1-y2), (x2-x1), (x1*y2 - x2*y1)
    denom = math.hypot(a,b)
    if denom == 0: return 0.0
    return (a*x0 + b*y0 + c) / denom

def classify_zone_from_distance(d):
    # d < 0 -> outside side; d >=0 inside side
    if d < 0:
        if abs(d) > NEAR_BOUNDARY_THRESHOLD:
            return "NEAR_BOUNDARY"
        elif abs(d) < CLIMBING_THRESHOLD:
            return "CLIMBING_OUTSIDE"
        else:
            return "OUTSIDE"
    else:
        if abs(d) < CLIMBING_THRESHOLD:
            return "CLIMBING_INSIDE"
        else:
            return "INSIDE"

# ---------------- Simple IoU tracker (greedy) ----------------
def iou(b1, b2):
    xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    a1 = max(1e-6, (b1[2]-b1[0])*(b1[3]-b1[1]))
    a2 = max(1e-6, (b2[2]-b2[0])*(b2[3]-b2[1]))
    return inter / (a1 + a2 - inter)

class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_lost=30):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.tracks = {}   # id -> dict {bbox, last_seen, lost, sustained_neg_frames, history deque}
        self.next_id = 1
    def update(self, detections, frame_idx):
        # detections: list of (x1,y1,x2,y2,conf)
        assigned = {}
        det_boxes = [d[:4] for d in detections]
        if len(self.tracks) > 0 and len(det_boxes) > 0:
            track_ids = list(self.tracks.keys())
            costs = np.ones((len(track_ids), len(det_boxes)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                for j, db in enumerate(det_boxes):
                    costs[i,j] = 1.0 - iou(self.tracks[tid]['bbox'], db)
            while True:
                i,j = np.unravel_index(np.argmin(costs), costs.shape)
                if costs[i,j] > 1.0 - self.iou_thresh:
                    break
                tid = track_ids[i]
                self.tracks[tid]['bbox'] = det_boxes[j]
                self.tracks[tid]['last_seen'] = frame_idx
                self.tracks[tid]['lost'] = 0
                self.tracks[tid]['history'].append((frame_idx, det_boxes[j]))
                assigned[j] = tid
                costs[i,:] = 1.0
                costs[:,j] = 1.0
        # create new tracks for unassigned detections
        for j, det in enumerate(detections):
            if j in assigned: continue
            box = det[:4]
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {'bbox': box, 'last_seen': frame_idx, 'lost': 0,
                                'history': deque(maxlen=10),
                                'sustained_neg': 0, 'last_label': None}
            self.tracks[tid]['history'].append((frame_idx, box))
            assigned[j] = tid
        # increment lost for tracks not updated
        remove = []
        for tid, t in list(self.tracks.items()):
            if t['last_seen'] < frame_idx:
                t['lost'] += 1
            if t['lost'] > self.max_lost:
                remove.append(tid)
        for tid in remove:
            del self.tracks[tid]
        return assigned

# ---------------- Process all videos ----------------
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
    CONFIRM_FRAMES = max(3, int(round(SUSTAIN_SECONDS * fps)))  # per-track confirmation frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(output_folder, os.path.splitext(vid_name)[0] + "_annotated.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

    # get first frame for line selection
    ret, first_frame = cap.read()
    if not ret:
        print("⚠️ Could not read first frame."); cap.release(); writer.release(); continue
    disp, scale = resize_for_display(first_frame, resize_max_side)
    line_choice = get_line_from_user(disp, window_title=f"Draw line for {vid_name}")
    if line_choice == "QUIT":
        print("User quit."); cap.release(); writer.release(); break
    if line_choice is None:
        print("Skipped video:", vid_name); cap.release(); writer.release(); continue
    p1_disp, p2_disp = line_choice
    if scale != 1.0:
        p1_orig = (p1_disp[0]/scale, p1_disp[1]/scale)
        p2_orig = (p2_disp[0]/scale, p2_disp[1]/scale)
    else:
        p1_orig, p2_orig = p1_disp, p2_disp

    # save clicked line
    json_line_path = os.path.join(output_folder, os.path.splitext(vid_name)[0] + "_line.json")
    with open(json_line_path, "w") as jf:
        json.dump({"p1_disp": p1_disp, "p2_disp": p2_disp, "p1_orig": p1_orig, "p2_orig": p2_orig, "scale": scale}, jf)

    # rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rows = []
    frame_id = 0
    start_time = time.time()

    tracker = SimpleTracker(iou_thresh=0.35, max_lost= max(30, int(fps*3)))

    print(f"Processing frames (confirm frames = {CONFIRM_FRAMES}). Press 'q' in preview window to stop early.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # resize for inference/display
        if scale != 1.0:
            disp = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
        else:
            disp = frame.copy()
        H_disp, W_disp = disp.shape[:2]
        image_area = W_disp * H_disp

        # inference
        res = model(disp, conf=conf_thresh, verbose=False)[0]
        annotated_disp = disp.copy()
        cv2.line(annotated_disp, p1_disp, p2_disp, (255,0,0), 2)

        detections = []
        if hasattr(res, "boxes") and len(res.boxes) > 0:
            for b in res.boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                names = res.names
                cls_name = names.get(cls_id, str(cls_id))
                if cls_name.lower() != "person": continue
                # bbox coords
                xy = b.xyxy[0].cpu().numpy()
                x1,y1,x2,y2 = [float(v) for v in xy]
                # spatial filters
                area = max(1.0, (x2-x1)*(y2-y1))
                if area < MIN_BBOX_AREA_RATIO * image_area: 
                    continue
                h = max(1.0, y2-y1); w = max(1.0, x2-x1)
                ar = h / w
                if ar < ASPECT_RATIO_MIN or ar > ASPECT_RATIO_MAX:
                    continue
                if conf < MIN_CONF:
                    continue
                # keep detection
                detections.append((x1,y1,x2,y2,conf))

        # tracker update
        assigned = tracker.update(detections, frame_id)  # mapping: det_index -> track_id

        # Count people in near zone this frame (to apply crowd rule)
        people_in_near_zone = 0
        # First pass: compute distances and temporary labels
        det_tmp = {}
        for j, det in enumerate(detections):
            bbox = det[:4]
            cx, cy = centroid(bbox)
            d = signed_distance((cx,cy), p1_disp, p2_disp)
            zone = classify_zone_from_distance(d)
            if zone == "NEAR_BOUNDARY":
                people_in_near_zone += 1
            det_tmp[j] = {'bbox': bbox, 'conf': det[4], 'd': d, 'zone': zone}

        # Now update tracks labels and sustained counts
        for det_idx, tid in assigned.items():
            info = det_tmp.get(det_idx)
            if info is None:
                continue
            bbox = info['bbox']; d = info['d']; zone = info['zone']
            t = tracker.tracks[tid]
            # update sustained negative-distance frames if outside side (d < 0)
            if d < 0:
                t['sustained_neg'] = t.get('sustained_neg', 0) + 1
            else:
                t['sustained_neg'] = 0
            t['last_label'] = zone
            # store latest bbox
            t['bbox'] = bbox

        # Decide effective confirm frames (crowd rule)
        confirm_frames_req = CONFIRM_FRAMES * (2 if people_in_near_zone > CROWD_MAX else 1)

        # draw detections and write rows
        det_count = 0
        for det_idx, tid in assigned.items():
            info = det_tmp[det_idx]
            bbox = info['bbox']; d = info['d']; zone = info['zone']; conf = info['conf']
            det_count += 1
            # per-track state
            t = tracker.tracks[tid]
            sustained = t.get('sustained_neg', 0)
            # decide final label: if sustained negative distance for >= confirm_frames_req => CROSSED (intrusion)
            final_label = zone
            alarm = False
            if zone.startswith("CLIMBING") and d < 0:
                # treat as potential climbing from outside; require sustained to call intrusion
                if sustained >= confirm_frames_req:
                    final_label = "CROSSED"
                    alarm = True
                else:
                    final_label = "CLIMBING"
            elif zone == "NEAR_BOUNDARY":
                # NEAR_BOUNDARY is ambiguous: require sustained negative to escalate
                if sustained >= confirm_frames_req:
                    final_label = "CROSSED"
                    alarm = True
            elif zone == "INSIDE":
                # inside + very close -> CLIMBING_INSIDE; optionally escalate if sustained
                if zone == "INSIDE" and abs(d) < CLIMBING_THRESHOLD:
                    if sustained >= max(1, int(0.5 * CONFIRM_FRAMES)):
                        final_label = "CROSSED"
                        alarm = True
                    else:
                        final_label = "CLIMBING_INSIDE"

            # draw
            x1,y1,x2,y2 = map(int, bbox)
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            color = (0,255,0)  # default green
            if final_label in ("CLIMBING", "CLIMBING_INSIDE"):
                color = (0,165,255)  # orange-ish
            if final_label == "NEAR_BOUNDARY":
                color = (0,255,255)  # yellow
            if final_label == "CROSSED":
                color = (0,0,255)  # red (alarm)

            cv2.rectangle(annotated_disp, (x1,y1), (x2,y2), color, 2)
            cv2.circle(annotated_disp, (cx,cy), 3, (255,255,0), -1)
            txt = f"{final_label} d={d:.1f} s={sustained}"
            cv2.putText(annotated_disp, txt, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # map back to original coords for CSV
            if scale != 1.0:
                x1o,y1o,x2o,y2o = x1/scale, y1/scale, x2/scale, y2/scale
                cxo, cyo = cx/scale, cy/scale
                d_orig = d / scale
            else:
                x1o,y1o,x2o,y2o,cxo,cyo,d_orig = x1,y1,x2,y2,cx,cy,d

            rows.append({
                "video": vid_name, "frame": frame_id, "track_id": tid,
                "x1": x1o, "y1": y1o, "x2": x2o, "y2": y2o,
                "conf": float(conf), "label": final_label,
                "d_display": float(d), "d_orig": float(d_orig),
                "centroid_x": float(cxo), "centroid_y": float(cyo),
                "sustained_neg_frames": int(sustained), "alarm": bool(alarm)
            })

        # write frame to output (upscale if required)
        if scale != 1.0:
            annotated_full = cv2.resize(annotated_disp, (orig_w, orig_h))
        else:
            annotated_full = annotated_disp
        writer.write(annotated_full)

        # preview and early quit
        cv2.imshow("Processing...", cv2.resize(annotated_full, (min(960,orig_w), min(540,orig_h))))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped early by user.")
            break

        # occasional progress
        if frame_id % 200 == 0:
            elapsed = time.time() - start_time
            print(f"Processed frames: {frame_id}  elapsed: {elapsed:.1f}s  avg fps: {frame_id/elapsed:.1f}")

    # cleanup per-video
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # save CSV and confirm files
    csv_path = os.path.join(output_folder, os.path.splitext(vid_name)[0] + "_detections.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"✅ Done: {vid_name}")
    print(f"   → Annotated video: {out_path}")
    print(f"   → Detections CSV:  {csv_path}")
    print(f"   → Clicked line JSON: {json_line_path}")

print("\nAll videos processed.")
"""
Multi-video perimeter intrusion detector
— Uses YOLO11L + built-in ByteTrack for person tracking
— User draws a wall top line once per video
— Classifies each person as OUTSIDE, NEAR_BOUNDARY, CLIMBING, INSIDE, or CROSSED
— Saves annotated video, CSV logs, and line JSON
"""

import os, sys, math, time, json
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------- USER CONFIG ----------------
input_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/NewTestData"
output_folder = "/Users/singhharshvardhan580/Developer/Perimeter Detection/Output-ByteTrack"
model_name = "yolo11l.pt"
conf_thresh = 0.30
resize_max_side = 960
CLIMBING_THRESHOLD = 20
NEAR_BOUNDARY_THRESHOLD = 60
SUSTAIN_SECONDS = 0.6   # for CROSSED confirmation
# ------------------------------------------------

os.makedirs(output_folder, exist_ok=True)

print(f"Loading YOLO model: {model_name}")
model = YOLO(model_name)
print("✅ Model loaded with ByteTrack tracking available.")

# ---------------- Mouse logic ----------------
CLICK_POINTS = []
def mouse_cb(event, x, y, flags, param):
    global CLICK_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINTS.append((int(x), int(y)))

def get_line_from_user(img_disp, window_title="Draw the wall top line (2 points)"):
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
            msg = "Click 2 points on the wall line. 's' skip, 'q' quit"
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

# ---------------- Geometry ----------------
def resize_for_display(img, max_side):
    if max_side is None: return img, 1.0
    h,w = img.shape[:2]
    if max(h,w)<=max_side: return img,1.0
    s=max_side/float(max(h,w)); new=(int(w*s),int(h*s))
    return cv2.resize(img,new),s

def centroid(b): x1,y1,x2,y2=b; return ((x1+x2)/2,(y1+y2)/2)
def signed_distance(pt,p1,p2):
    x0,y0=pt; x1,y1=p1; x2,y2=p2
    a,b,c=(y1-y2),(x2-x1),(x1*y2-x2*y1)
    denom=math.hypot(a,b)
    return 0 if denom==0 else (a*x0+b*y0+c)/denom

def classify_zone(bbox,p1,p2):
    cx,cy=centroid(bbox)
    d=signed_distance((cx,cy),p1,p2)
    if d<0:
        if abs(d)>NEAR_BOUNDARY_THRESHOLD: label,color="NEAR_BOUNDARY",(0,255,255)
        elif abs(d)<CLIMBING_THRESHOLD: label,color="CLIMBING",(0,0,255)
        else: label,color="OUTSIDE",(0,255,0)
    else:
        if abs(d)<CLIMBING_THRESHOLD: label,color="CLIMBING",(0,165,255)
        else: label,color="INSIDE",(0,255,0)
    return label,color,d

# ---------------- Process videos ----------------
videos=[v for v in os.listdir(input_folder) if v.lower().endswith(".mp4")]
if not videos:
    print("No .mp4 found in:",input_folder); sys.exit(0)
print(f"Found {len(videos)} videos.\n")

for idx,vid in enumerate(videos,1):
    in_path=os.path.join(input_folder,vid)
    cap=cv2.VideoCapture(in_path)
    if not cap.isOpened(): print("⚠️ Can't open",vid); continue
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CONFIRM_FRAMES=max(3,int(SUSTAIN_SECONDS*fps))

    # first frame to draw line
    ret,first=cap.read()
    if not ret: cap.release(); continue
    disp,scale=resize_for_display(first,resize_max_side)
    lc=get_line_from_user(disp,f"Draw wall top for {vid}")
    if lc=="QUIT": break
    if lc is None: cap.release(); continue
    p1_disp,p2_disp=lc
    p1_orig=(p1_disp[0]/scale,p1_disp[1]/scale); p2_orig=(p2_disp[0]/scale,p2_disp[1]/scale)
    json_line_path=os.path.join(output_folder,os.path.splitext(vid)[0]+"_line.json")
    with open(json_line_path,"w") as jf:
        json.dump({"p1_disp":p1_disp,"p2_disp":p2_disp,"p1_orig":p1_orig,"p2_orig":p2_orig,"scale":scale},jf)

    # prepare writer
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    out_path=os.path.join(output_folder,os.path.splitext(vid)[0]+"_annotated.mp4")
    writer=cv2.VideoWriter(out_path,fourcc,fps,(w,h))

    # tracking via ByteTrack
    track_results=[]
    print(f"[{idx}/{len(videos)}] Processing {vid} ... Press 'q' to stop early.")
    frame_id=0
    rows=[]
    sustained_tracker={}  # track_id -> sustained d<0 frames

    while True:
        ret,frame=cap.read()
        if not ret: break
        frame_id+=1
        if scale!=1.0:
            disp=cv2.resize(frame,(int(frame.shape[1]*scale),int(frame.shape[0]*scale)))
        else: disp=frame.copy()

        # run YOLO tracking (ByteTrack)
        results=model.track(source=disp, persist=True, tracker="bytetrack.yaml",
                            conf=conf_thresh, verbose=False)
        if not results: break
        res=results[0]
        annotated=disp.copy()
        cv2.line(annotated,p1_disp,p2_disp,(255,0,0),2)

        if hasattr(res,"boxes") and len(res.boxes)>0:
            for b in res.boxes:
                if int(b.cls[0])!=0: continue  # class 0 = person
                x1,y1,x2,y2=b.xyxy[0].cpu().numpy()
                bbox=(x1,y1,x2,y2)
                track_id=int(b.id[0]) if b.id is not None else -1
                conf=float(b.conf[0])
                label,color,d=classify_zone(bbox,p1_disp,p2_disp)

                # sustained logic
                if track_id!=-1:
                    sustained_tracker.setdefault(track_id,0)
                    if d<0: sustained_tracker[track_id]+=1
                    else: sustained_tracker[track_id]=0
                    if sustained_tracker[track_id]>=CONFIRM_FRAMES:
                        label="CROSSED"; color=(0,0,255)

                # draw
                x1i,y1i,x2i,y2i=int(x1),int(y1),int(x2),int(y2)
                cx,cy=int((x1+x2)/2),int((y1+y2)/2)
                cv2.rectangle(annotated,(x1i,y1i),(x2i,y2i),color,2)
                cv2.circle(annotated,(cx,cy),3,(255,255,0),-1)
                cv2.putText(annotated,f"ID{track_id} {label} d={d:.1f}",
                            (x1i,max(16,y1i-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)

                # record
                rows.append({
                    "video":vid,"frame":frame_id,"track_id":track_id,"label":label,
                    "x1":x1/scale,"y1":y1/scale,"x2":x2/scale,"y2":y2/scale,
                    "centroid_x":cx/scale,"centroid_y":cy/scale,
                    "distance_disp":d,"conf":conf,
                    "sustained_frames":sustained_tracker.get(track_id,0)
                })

        annotated_full=cv2.resize(annotated,(w,h)) if scale!=1.0 else annotated
        writer.write(annotated_full)
        cv2.imshow("Processing...",cv2.resize(annotated_full,(min(960,w),min(540,h))))
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); writer.release(); cv2.destroyAllWindows()

    csv_path=os.path.join(output_folder,os.path.splitext(vid)[0]+"_detections.csv")
    pd.DataFrame(rows).to_csv(csv_path,index=False)
    print(f"✅ {vid}: saved → {out_path}")
    print(f"   → CSV:  {csv_path}")
    print(f"   → Line: {json_line_path}")

print("\nAll videos processed.")
