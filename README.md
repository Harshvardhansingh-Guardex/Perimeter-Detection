#test6.py is the finalised code 

# YOLO Perimeter Detection System (BBox-Overlap Method)

## Overview

This system processes videos to detect people and determine whether they are **ON_GROUND** (inside a defined perimeter/region) or **OFF_GROUND** (outside the region). It uses bounding box intersection with a user-drawn ground polygon to make this classification, making it ideal for perimeter security, restricted area monitoring, and spatial compliance tracking.

## What It Does

The algorithm:
1. **Loads videos** from an input folder (batch processing)
2. **Interactive region drawing** - User draws a ground polygon on the first frame
3. **Detects & tracks people** using YOLO11 + ByteTrack
4. **Computes overlap** - Calculates what fraction of each person's bounding box overlaps with the ground polygon
5. **Classifies location** - Labels as ON_GROUND or OFF_GROUND based on overlap threshold
6. **Outputs** annotated videos, CSV logs, and region definitions (JSON)

## Core Algorithm: Bounding Box Overlap

### How Location is Determined

For each detected person, the system computes:

```
bbox_overlap_fraction = intersection_area / bbox_area
```

Where:
- **intersection_area** = Area where bounding box overlaps with ground polygon
- **bbox_area** = Total area of the person's bounding box

### Classification Logic

```
IF bbox_overlap_fraction >= BBOX_FRAC_THRESH:
    Label = ON_GROUND (person is inside/on the ground region)
ELSE:
    Label = OFF_GROUND (person is outside the ground region)
```

### Why This Works

- **Size-invariant**: Uses fraction (0-1) rather than absolute pixels
- **Partial overlap handling**: Detects people partially on/off ground
- **Robust to pose**: Works regardless of whether person is standing, sitting, lying down
- **Simple geometry**: Fast intersection calculation using Shapely library

## Pipeline Architecture

```
┌──────────────────────┐
│  Load Videos (.mp4)  │
│  from INPUT_FOLDER   │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Read First Frame                   │
│  - Display for user interaction     │
│  - Optional: Resize for display     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Interactive Polygon Drawing        │
│  - Left-click: Add polygon point    │
│  - Right-click: Finalize polygon    │
│  - User defines ground region       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Optional Wall Line Drawing         │
│  - Click 2 points for reference line│
│  - Press 's' to skip                │
│  - Not used in classification       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Save Region Definition             │
│  - JSON: polygon coords, line, scale│
│  - Enables region reuse/audit       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Frame-by-Frame Processing          │
│  ┌─────────────────────────────┐   │
│  │ YOLO Detection (class=person)│   │
│  │ - Confidence > CONF_THRESH   │   │
│  └─────────────┬───────────────┘   │
│                ▼                    │
│  ┌─────────────────────────────┐   │
│  │ ByteTrack Tracking           │   │
│  │ - Assign persistent track IDs│   │
│  └─────────────┬───────────────┘   │
│                ▼                    │
│  ┌─────────────────────────────┐   │
│  │ Compute BBox Overlap         │   │
│  │ - Intersection with polygon  │   │
│  │ - Calculate overlap fraction │   │
│  └─────────────┬───────────────┘   │
│                ▼                    │
│  ┌─────────────────────────────┐   │
│  │ Classify Location            │   │
│  │ - ON_GROUND vs OFF_GROUND    │   │
│  │ - Based on BBOX_FRAC_THRESH  │   │
│  └─────────────┬───────────────┘   │
│                ▼                    │
│  ┌─────────────────────────────┐   │
│  │ Annotate Frame               │   │
│  │ - Draw filled polygon        │   │
│  │ - Draw bbox (color-coded)    │   │
│  │ - Draw centroid              │   │
│  │ - Add label with track ID    │   │
│  └─────────────┬───────────────┘   │
│                ▼                    │
│  ┌─────────────────────────────┐   │
│  │ Log Detection                │   │
│  │ - CSV row with all metrics   │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Output Generation                  │
│  - Annotated video (full resolution)│
│  - CSV with all detections          │
│  - Regions JSON (already saved)     │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Process Next Video                 │
│  (Repeat for all videos in folder)  │
└─────────────────────────────────────┘
```

## Configuration Parameters

| Parameter | Default | Description | Effect of Increasing | Effect of Decreasing |
|-----------|---------|-------------|---------------------|---------------------|
| `CONF_THRESH` | 0.30 | YOLO detection confidence | Fewer false positives, may miss people | More detections, more false positives |
| `BBOX_FRAC_THRESH` | 0.01 | Overlap fraction for ON_GROUND | Stricter (more overlap needed) | Looser (less overlap needed) |
| `RESIZE_MAX_SIDE` | 960 | Display resolution for processing | Better quality, slower processing | Faster processing, lower quality |
| `FILL_ALPHA` | 0.45 | Transparency of filled polygon | More opaque overlay | More transparent overlay |
| `MODEL_NAME` | "yolo11n.pt" | YOLO model variant | Larger models (s/m/l) = more accurate but slower | Smaller = faster but less accurate |

### Critical Threshold: `BBOX_FRAC_THRESH`

This is the **most important parameter** for tuning detection behavior.

#### Understanding the Threshold

The threshold represents what **fraction of a person's bounding box** must overlap with the ground polygon to be classified as ON_GROUND.

```
bbox_overlap_fraction = (intersection_area) / (total_bbox_area)
```

#### Threshold Values and Their Effects

| Threshold | Meaning | Use Case | Pros | Cons |
|-----------|---------|----------|------|------|
| **0.01** (1%) | Very sensitive | Detect anyone touching ground | Catches partial intrusions | Many false positives |
| **0.10** (10%) | Lenient | Security with buffer zone | Balanced detection | May miss edge cases |
| **0.30** (30%) | Moderate | Standard perimeter | Good accuracy | Person must be mostly inside |
| **0.50** (50%) | Strict | Restricted area compliance | High confidence | May miss valid intrusions |
| **0.70+** (70%+) | Very strict | Full containment required | Very few false positives | Will miss many valid cases |

#### Visual Examples

**Scenario: Person at edge of ground polygon**

```
Threshold = 0.10 (10%)
┌─────────────────┐
│   Person BBox   │  ← Only 15% overlaps with ground
│        ▓▓▓▓▓▓▓▓▓│  → Result: ON_GROUND ✓
└────────░░░░░░░░░┘
         Ground Polygon
```

```
Threshold = 0.30 (30%)
┌─────────────────┐
│   Person BBox   │  ← Only 15% overlaps with ground
│        ▓▓▓▓▓▓▓▓▓│  → Result: OFF_GROUND ✗
└────────░░░░░░░░░┘
         Ground Polygon
```

#### Recommended Settings by Use Case

**Security/Intrusion Detection (Catch everything)**
```python
BBOX_FRAC_THRESH = 0.05  # 5% - very sensitive
```
- **Why**: Better to have false alarms than miss an intrusion
- **Tradeoff**: People walking near boundary may trigger alerts

**Restricted Area Monitoring (Balanced)**
```python
BBOX_FRAC_THRESH = 0.20  # 20% - moderate
```
- **Why**: Good balance between accuracy and false positives
- **Tradeoff**: Occasional edge cases may be misclassified

**Compliance/Safety Zones (High confidence)**
```python
BBOX_FRAC_THRESH = 0.40  # 40% - strict
```
- **Why**: Only flag when person is clearly inside zone
- **Tradeoff**: May miss people just stepping into zone

**Full Containment (Very strict)**
```python
BBOX_FRAC_THRESH = 0.70  # 70% - very strict
```
- **Why**: Verify person is almost entirely inside zone
- **Tradeoff**: Will miss many valid detections at edges

### Other Important Parameters

#### `RESIZE_MAX_SIDE`

Controls processing resolution (not output resolution - output is always full resolution).

- **960 (default)**: Good balance, ~30-60 FPS on GPU
- **1280**: Higher quality, ~20-40 FPS on GPU
- **None**: Native resolution, ~10-20 FPS on GPU, highest quality
- **640**: Fast processing, ~60-120 FPS on GPU, lower quality

**Note**: Detection happens at display scale, but coordinates are mapped back to original resolution for output.

#### `CONF_THRESH`

YOLO detection confidence threshold.

- **0.20-0.25**: Catch more people, more false positives (shadows, reflections)
- **0.30-0.40**: Balanced (recommended)
- **0.50+**: Only high-confidence detections, may miss people in poor lighting


## Interactive Region Drawing

### Polygon Drawing (Ground Region)

**Controls:**
- **Left-click**: Add a point to polygon
- **Right-click**: Finalize polygon (minimum 3 points required)
- **'q' key**: Cancel and skip (will mark all as OFF_GROUND)

**Best Practices:**
1. **Define clear boundaries**: Click points along the actual ground perimeter
2. **Close the shape**: Polygon auto-closes from last point to first
3. **Use enough points**: 4-8 points usually sufficient, more for curved boundaries
4. **Avoid self-intersecting polygons**: Keep shape simple and convex if possible

**Visual Feedback:**
- Yellow dots show clicked points
- Yellow outline shows polygon edges
- Dark blue fill (45% transparent) shows active ground region

### Optional Wall Line

**Purpose**: Reference line (not used in classification, purely visual)

**Controls:**
- **Click twice**: Define two endpoints
- **'s' key**: Skip (recommended unless you need visual reference)
- **'r' key**: Reset points
- **'c' key**: Confirm
- **'q' key**: Quit entire process

## Output Files

### 1. Annotated Video (`*_annotated.mp4`)

**Visual Elements:**
- 🔵 **Dark blue filled polygon**: Ground region (semi-transparent)
- 🟡 **Yellow outline**: Ground polygon boundary
- 🟢 **Green bounding box**: Person classified as ON_GROUND
- 🔴 **Red bounding box**: Person classified as OFF_GROUND
- 🟡 **Yellow dot**: Person's centroid (center of bounding box)
- **Label format**: `ID{track_id} {ON_GROUND|OFF_GROUND} {overlap_fraction}`

**Example label**: `ID42 ON_GROUND 0.65` means:
- Track ID 42
- Inside ground region
- 65% of bounding box overlaps with ground

### 2. Detections CSV (`*_detections.csv`)

Contains one row per detection per frame:

```csv
video,frame,track_id,label,bbox_overlap_frac,inter_area,bbox_area,x1,y1,x2,y2,centroid_x,centroid_y
video1.mp4,1,1,ON_GROUND,0.65,12500.5,19231.0,320.2,180.5,450.7,520.3,385.45,350.4
video1.mp4,2,1,ON_GROUND,0.63,12100.2,19231.0,322.1,182.3,451.0,521.5,386.55,351.9
```

**Column Descriptions:**

| Column | Type | Description |
|--------|------|-------------|
| `video` | string | Source video filename |
| `frame` | int | Frame number (1-indexed) |
| `track_id` | int | Persistent person ID from ByteTrack |
| `label` | string | ON_GROUND or OFF_GROUND |
| `bbox_overlap_frac` | float | Fraction of bbox overlapping ground (0-1) |
| `inter_area` | float | Intersection area in pixels² |
| `bbox_area` | float | Total bounding box area in pixels² |
| `x1, y1, x2, y2` | float | Bounding box corners (original resolution) |
| `centroid_x, centroid_y` | float | Bounding box center (original resolution) |



### 3. Regions JSON (`*_regions.json`)

Stores the user-defined regions for reproducibility:

```json
{
  "p1": [100, 200],
  "p2": [800, 200],
  "ground_pts": [
    [150, 300],
    [750, 300],
    [700, 600],
    [200, 600]
  ],
  "scale": 0.75
}
```

**Fields:**
- `p1, p2`: Optional wall line endpoints (display coordinates)
- `ground_pts`: List of polygon vertices (display coordinates)
- `scale`: Display scale factor used during drawing (for coordinate mapping)

**Use**: Re-run analysis with same region, audit region definitions, batch processing with pre-defined regions

## Installation

```bash
# Install required packages
pip install ultralytics opencv-python pandas shapely

# Download YOLO model (auto-downloads on first run)
# Models: yolo11n.pt (nano), yolo11s.pt (small), yolo11m.pt (medium), yolo11l.pt (large)
```


## Usage

### Basic Workflow

1. **Organize videos**:
```bash
mkdir input_videos output_results
# Place all .mp4 files in input_videos/
```

2. **Configure script**:
```python
INPUT_FOLDER = "./input_videos"
OUTPUT_FOLDER = "./output_results"
MODEL_NAME = "yolo11n.pt"  # or yolo11s.pt, yolo11m.pt, yolo11l.pt
BBOX_FRAC_THRESH = 0.20    # Adjust based on use case (see table above)
```

3. **Run**:
```bash
python perimeter_detector.py
```

4. **Interactive steps for each video**:
   - **Step 1**: Optional wall line (press 's' to skip)
   - **Step 2**: Draw ground polygon (left-click points, right-click finish)
   - **Step 3**: Processing begins (live preview shown)

5. **Review outputs**:
   - Watch annotated videos
   - Analyze CSV files for metrics
   - Check regions JSON for region definitions


## Use Case & Recommended Settings

### 1. Perimeter Security (Fence/Boundary)

**Goal**: Detect anyone crossing into restricted area

```python
MODEL_NAME = "yolo11s.pt"     # Fast + accurate
CONF_THRESH = 0.25            # Catch more detections
BBOX_FRAC_THRESH = 0.05       # Very sensitive (5%)
RESIZE_MAX_SIDE = 1280        # Higher resolution
```

**Drawing Strategy**: Draw polygon just inside the fence/boundary
**Expected Behavior**: Alert on any person with 5%+ of body inside area


**Drawing Strategy**: Draw polygon around excavation, heavy machinery areas
**Expected Behavior**: Alert on any intrusion into danger zone





## Technical Notes

- **Coordinate system**: Top-left origin (0,0), x-right, y-down
- **Tracker persistence**: `persist=True` maintains IDs across frames
- **Resolution handling**: Detection at display scale, output at original scale
- **Shapely geometry**: Uses robust computational geometry library
- **Frame duplication**: Not used here (unlike stillness detector), outputs at original FPS


---

**Quick Start Checklist:**
- [ ] Install dependencies: `pip install ultralytics opencv-python pandas shapely`
- [ ] Place videos in `INPUT_FOLDER`
- [ ] Adjust `BBOX_FRAC_THRESH` for your use case (see table)
- [ ] Run script, draw polygons interactively
- [ ] Review annotated videos and CSV outputs
- [ ] Fine-tune thresholds based on results
