# AI-Based Workplace Safety Monitoring System

Real-time PPE compliance and restricted-zone monitoring for construction sites
and industrial workplaces, built with **YOLOv8 + DeepSORT + OpenCV**.

```
Camera / Video
      |
   OpenCV
      |
   YOLOv8
      |
Person Detection + PPE Detection
      |
PPE-to-Person Matching
      |
DeepSORT Worker Tracking
      |
PPE Compliance Analysis
      |
Restricted Zone Monitoring
      |
Violation Detection
      |
Logs + Evidence Snapshots
```

---

## Features

- **Person detection** with YOLOv8 (COCO class `person`)
- **Helmet / Vest / Gloves detection** with your own trained YOLOv8 PPE model
- **PPE-to-person matching** — PPE is never counted globally; each item is
  assigned to exactly one worker using containment + body-region logic
- **Worker tracking** with DeepSORT → stable IDs (`Worker #12`)
- **PPE compliance classification**: `SAFE` / `PARTIAL` / `UNSAFE`
- **Temporal smoothing** so one missed frame never flips a worker's status
- **Restricted-zone monitoring** with mouse-drawn polygons and
  `cv2.pointPolygonTest()` on the worker's ground point
- **Violation detection** with persistence + cooldown (no per-frame spam)
- **Annotated evidence snapshots** saved automatically
- **CSV + JSON violation logs**
- **Live dashboard overlay** (workers, safe/partial/unsafe, compliance %, FPS)
- **Processed output video**

## Technologies

Python 3.10+ · YOLOv8 (Ultralytics) · DeepSORT (`deep-sort-realtime`) ·
OpenCV · PyTorch · NumPy · Pandas · PyYAML

---

## Project structure

```
workplace_safety_ai/
├── main.py                  # entry point + frame loop
├── requirements.txt
├── README.md
├── config.yaml              # all tunable settings
│
├── models/
│   ├── person_model/        # (optional) local copy of yolov8n.pt
│   └── ppe_model/
│       └── best.pt          # <-- YOUR trained PPE weights go here
│
├── src/
│   ├── __init__.py
│   ├── detector.py          # YOLOv8 person + PPE detectors
│   ├── tracker.py           # DeepSORT wrapper -> TrackedWorker
│   ├── ppe_matcher.py       # PPE -> person assignment
│   ├── compliance.py        # smoothing + SAFE/PARTIAL/UNSAFE
│   ├── restricted_zone.py   # polygon zones + intrusion test
│   ├── violation_manager.py # debounce, cooldown, event building
│   ├── logger.py            # CSV/JSON logs + evidence snapshots
│   ├── visualizer.py        # every OpenCV overlay
│   └── utils.py             # config, device, geometry, FPS
│
├── videos/
│   └── input.mp4            # <-- YOUR test video goes here
│
├── output/
│   ├── processed/           # result.mp4
│   ├── snapshots/           # evidence images
│   └── logs/                # violations.csv / violations.json
│
├── data/
│   └── zones.json           # saved restricted-zone polygons
└── runs/                    # YOLO scratch dir
```

---

## Installation (Windows)

1. Install **Python 3.10 or 3.11** (tick *Add Python to PATH*).
2. Open **Command Prompt** and go to the project folder:

```cmd
cd path\to\workplace_safety_ai
```

3. Create a virtual environment:

```cmd
python -m venv venv
```

4. Activate it:

```cmd
venv\Scripts\activate
```

5. Install dependencies:

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional, NVIDIA GPU)* install the CUDA build of PyTorch first:

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

6. Put your trained PPE model here:

```
models\ppe_model\best.pt
```

7. Put your video here:

```
videos\input.mp4
```

8. Run it:

```cmd
python main.py --source videos/input.mp4
```

Webcam:

```cmd
python main.py --source 0
```

Other useful flags:

```cmd
python main.py --source videos/site.mkv --select-zone   :: redraw the zone
python main.py --source videos/input.mp4 --no-display   :: headless / faster
python main.py --source videos/input.mp4 --cpu          :: force CPU
python main.py --source videos/input.mp4 --no-zone      :: PPE only
```

`yolov8n.pt` (person detector) downloads automatically on first run, so the
first launch needs an internet connection.

---

## Where the models and video must be placed

| What | Exact location | Notes |
|---|---|---|
| PPE weights | `models/ppe_model/best.pt` | Your own YOLOv8 model trained on helmet / vest / gloves. Change the path in `config.yaml → models.ppe_model` if you keep it elsewhere. |
| Person weights | auto-downloaded `yolov8n.pt` | Any COCO YOLOv8 model works (`yolov8s.pt`, `yolov8m.pt`…). |
| Video | `videos/input.mp4` | Or pass any path with `--source`. `.mp4 .avi .mov .mkv` supported. |

**Important:** the default COCO YOLOv8 model **cannot** detect helmets, vests
or gloves — it only knows 80 generic COCO classes. If `best.pt` is missing the
app prints:

```
PPE model not found. Place your trained PPE YOLOv8 model at
models/ppe_model/best.pt or update config.yaml.
```

and keeps running (detection, tracking, zones, logging all still work) instead
of faking PPE results.

Free PPE datasets/models to train or download from: Roboflow Universe
("Hard Hat Workers", "Construction Site Safety", "PPE Detection"). After
training, copy `runs/detect/train/weights/best.pt` into `models/ppe_model/`.

If your model's class names differ (`Hardhat`, `Safety Vest`, `NO-Hardhat`…),
just list them under `ppe.class_aliases` in `config.yaml`. At startup the app
prints every class in your weights and how it was mapped.

---

## Restricted-zone selection

When the app starts (and no saved zone exists) a still frame opens so you can
draw the zone:

| Key / Mouse | Action |
|---|---|
| **Left click** | add a polygon point |
| **Right click** | close the current polygon (min. 3 points) |
| **ENTER** | close polygon, or finish if none pending |
| **R** | reset all polygons |
| **S** | save zones and start monitoring |
| **ESC** | skip (run without a zone) |

Zones are saved to `data/zones.json` and reloaded automatically next run.
Delete that file, press **R** during playback, or run with `--select-zone` to
redraw. You can also hard-code polygons in `config.yaml`:

```yaml
restricted_zone:
  polygons:
    - [[420, 300], [900, 300], [980, 620], [380, 620]]
```

A worker counts as inside when the **bottom-centre** of their tracked box
(their feet, i.e. their position on the ground) falls inside the polygon.

---

## Keyboard controls during playback

| Key | Action |
|---|---|
| **Q** or **ESC** | quit |
| **P** | pause / resume |
| **R** | reset & redraw the restricted zone |
| **S** | save a manual screenshot |

---

## Where results are stored

| Output | Path |
|---|---|
| Processed video | `output/processed/result.mp4` |
| Evidence snapshots | `output/snapshots/worker_12_missing_gloves_2026-09-19_10-35-21.jpg` |
| Violation log (CSV) | `output/logs/violations.csv` |
| Violation log (JSON) | `output/logs/violations.json` |
| Zone polygons | `data/zones.json` |

CSV columns:

```
event_id, timestamp, video_time, worker_id, violation_type,
helmet, vest, gloves, ppe_status, restricted_zone,
confidence, snapshot_path, source
```

Rows are **appended** across runs. Set `output.append_logs: false` in
`config.yaml` to start a fresh file each run.

---

## How compliance is decided

1. YOLOv8 detects persons and PPE items in the frame.
2. Each PPE box is scored against every tracked worker:
   `0.6 × containment + 0.3 × body-region fit + 0.1 × confidence`,
   and greedily assigned to its best worker (one box, one owner).
3. The raw per-frame result is pushed into a rolling history
   (`compliance.smoothing_frames`, default 15). An item counts as worn if it
   was seen in at least `presence_ratio` (default 35 %) of that history — this
   kills the `SAFE → UNSAFE → SAFE` flicker caused by single-frame misses.
4. Classification:
   - **SAFE** — all required items present
   - **PARTIAL** — some present, some missing
   - **UNSAFE** — nothing present, or ≥ `unsafe_if_missing_at_least` missing

Violations are only written when the condition survives
`violations.min_violation_frames` consecutive frames, and the same
(worker, violation) pair is muted for `violations.cooldown_seconds`
afterwards. If the worker becomes compliant and offends again later, the event
is logged immediately.

> Note: OpenCV's built-in fonts are ASCII-only, so the overlay shows `[+]` /
> `[-]` instead of ✓ / ✗ and `/!\` instead of ⚠. Everything else is identical.

---

## Demo script (for judges)

1. `python main.py --source videos/input.mp4`
2. Draw a restricted zone around a hazardous area → press **S**.
3. Point at a compliant worker: `Worker #3 … STATUS: SAFE` (green box).
4. Point at a worker without gloves: `Worker #7 … STATUS: PARTIAL` (yellow),
   then show the new row in `output/logs/violations.csv` and its snapshot.
5. Point at a worker with no PPE: `Worker #11 … STATUS: UNSAFE` (red).
6. Let someone walk into the polygon:
   `/!\ VIOLATION: WORKER #14 ENTERED RESTRICTED ZONE` + evidence saved.
7. Open `output/processed/result.mp4` — the whole session is recorded with
   every overlay.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `PPE model not found` | Copy your trained weights to `models/ppe_model/best.pt`, or update `models.ppe_model` in `config.yaml`. |
| PPE model loads but nothing is detected | Check the class-mapping table printed at startup. Unmapped classes need entries under `ppe.class_aliases`. Also lower `detection.ppe_confidence`. |
| `ModuleNotFoundError: ultralytics` / `deep_sort_realtime` | The venv isn't active or deps aren't installed: `venv\Scripts\activate` then `pip install -r requirements.txt`. |
| `Device: CPU` although you have a GPU | The CPU-only torch wheel is installed. Reinstall torch from the CUDA index URL above. Verify with `python -c "import torch;print(torch.cuda.is_available())"`. |
| Very low FPS | Use `yolov8n.pt`, lower `display.resize_width` (e.g. 960), run with `--no-display`, or use a GPU. |
| Camera won't open | Another app is using it, or try `--source 1`, `--source 2`. |
| Video opens but ends instantly | Unsupported codec — re-encode to H.264 MP4 (`ffmpeg -i in.mkv -c:v libx264 out.mp4`). |
| No window appears / `cv2.error … not implemented` | Headless OpenCV build or no desktop session. Run with `--no-display`; zones then come from `data/zones.json` or `config.yaml`. |
| DeepSORT downloads embedder weights on first run | Normal — needs internet once. |
| Worker IDs change too often | Increase `tracking.max_age`, lower `tracking.max_iou_distance`, or raise `detection.person_confidence`. |
| Too many log rows | Increase `violations.cooldown_seconds` / `min_violation_frames`, or switch off `ppe_partial` / `ppe_unsafe` under `violations.types` and keep only the per-item violations. |
| `output/processed/result.mp4` is empty | The writer couldn't open the codec — the console says so. Try a different filename with `--output result.avi`. |
| Zone appears in the wrong place | Zones are stored in the **resized** frame's pixel space. If you change `display.resize_width`, delete `data/zones.json` and redraw. |

---

## Configuration reference

Every threshold lives in `config.yaml`: model paths, confidences, DeepSORT
parameters, required PPE list and class aliases, matching thresholds,
smoothing window, violation cooldown, zone behaviour, output paths and device
selection. The app merges your file on top of built-in defaults, so a missing
or partially filled `config.yaml` never crashes the run.

---

*Built for a hackathon demonstration: DETECTION → TRACKING → ANALYSIS →
VIOLATION → EVIDENCE.*
