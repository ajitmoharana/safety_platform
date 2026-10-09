"""
Shared helpers: config loading, device selection, geometry and timing utilities.
Kept dependency-light so every other module can import it safely
(no module in src/ imports another src module from here -> no circular imports).
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path_like: str | Path) -> Path:
    """Resolve a (possibly relative) path against the project root.

    Using pathlib everywhere keeps Windows paths safe.
    """
    p = Path(str(path_like)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "models": {"person_model": "yolov8n.pt", "ppe_model": "models/ppe_model/best.pt"},
    "detection": {
        "person_confidence": 0.45,
        "ppe_confidence": 0.35,
        "iou_threshold": 0.45,
        "imgsz": 640,
    },
    "tracking": {
        "max_age": 30,
        "n_init": 3,
        "max_iou_distance": 0.7,
        "max_cosine_distance": 0.3,
        "embedder": "mobilenet",
        "half": False,
    },
    "ppe": {
        "required": ["helmet", "vest", "gloves"],
        "class_aliases": {
            "helmet": ["helmet", "hardhat", "hard hat"],
            "vest": ["vest", "safety vest"],
            "gloves": ["glove", "gloves"],
        },
        "ignore_classes": ["person"],
    },
    "matching": {
        "min_containment": 0.55,
        "regions": {"helmet": [0.0, 0.4], "vest": [0.15, 0.75], "gloves": [0.25, 0.95]},
        "out_of_region_penalty": 0.35,
    },
    "compliance": {
        "smoothing_frames": 15,
        "presence_ratio": 0.35,
        "min_history": 5,
        "unsafe_if_missing_at_least": 2,
    },
    "violations": {
        "cooldown_seconds": 8,
        "min_violation_frames": 5,
        "save_snapshot": True,
        "types": {
            "missing_helmet": True,
            "missing_vest": True,
            "missing_gloves": True,
            "ppe_partial": True,
            "ppe_unsafe": True,
            "restricted_zone_entry": True,
        },
    },
    "restricted_zone": {
        "enabled": True,
        "interactive_selection": True,
        "zone_file": "data/zones.json",
        "load_saved_zones": True,
        "polygons": [],
        "reference_point": "bottom_center",
    },
    "output": {
        "save_video": True,
        "save_logs": True,
        "save_snapshots": True,
        "processed_dir": "output/processed",
        "snapshots_dir": "output/snapshots",
        "logs_dir": "output/logs",
        "log_file": "output/logs/violations.csv",
        "append_logs": True,
        "output_video_name": "result.mp4",
    },
    "display": {
        "show_window": True,
        "show_ppe_boxes": True,
        "window_name": "Workplace Safety Monitor",
        "resize_width": 1280,
    },
    "device": {"use_cuda_if_available": True},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load config.yaml and merge it on top of safe defaults.

    Missing or invalid files never crash the app - defaults are used instead.
    """
    path = resolve_path(config_path)
    user_cfg: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                user_cfg = loaded
            else:
                print(f"[CONFIG] {path.name} is not a YAML mapping - using defaults.")
        except yaml.YAMLError as exc:
            print(f"[CONFIG] Could not parse {path}: {exc}")
            print("[CONFIG] Falling back to built-in default configuration.")
    else:
        print(f"[CONFIG] {path} not found - using built-in defaults.")
    return _deep_merge(DEFAULT_CONFIG, user_cfg)


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """Create every output directory the app writes to."""
    out = config.get("output", {})
    for key in ("processed_dir", "snapshots_dir", "logs_dir"):
        resolve_path(out.get(key, f"output/{key}")).mkdir(parents=True, exist_ok=True)
    resolve_path("data").mkdir(parents=True, exist_ok=True)
    resolve_path("videos").mkdir(parents=True, exist_ok=True)
    resolve_path("models/ppe_model").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def select_device(use_cuda_if_available: bool = True) -> str:
    """Return 'cuda' when a GPU is usable, otherwise 'cpu'."""
    if not use_cuda_if_available:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        print("[DEVICE] CUDA not available - falling back to CPU.")
    except Exception as exc:  # torch missing / broken install
        print(f"[DEVICE] Could not query CUDA ({exc}) - using CPU.")
    return "cpu"


def device_label(device: str) -> str:
    return "CUDA" if str(device).startswith("cuda") else "CPU"


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


def box_center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bottom_center(box: Box) -> Tuple[float, float]:
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Fraction of `inner` that lies inside `outer` (0..1)."""
    area = box_area(inner)
    if area <= 0:
        return 0.0
    return intersection_area(inner, outer) / area


def point_in_box(point: Sequence[float], box: Box) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def clamp_box(box: Box, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, x1)))
    y1 = int(max(0, min(height - 1, y1)))
    x2 = int(max(0, min(width - 1, x2)))
    y2 = int(max(0, min(height - 1, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def xyxy_to_ltwh(box: Box) -> List[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def ltwh_to_xyxy(box: Sequence[float]) -> Box:
    x, y, w, h = box
    return (float(x), float(y), float(x + w), float(y + h))


# --------------------------------------------------------------------------
# Text / naming helpers
# --------------------------------------------------------------------------
def normalise_label(label: str) -> str:
    """Lowercase a class name and strip separators so 'Hard-Hat' == 'hard hat'."""
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def safe_filename(text: str) -> str:
    keep = []
    for ch in str(text):
        keep.append(ch if (ch.isalnum() or ch in "-_.") else "_")
    return "".join(keep)


def seconds_to_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def timestamp_now(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return time.strftime(fmt)


def file_timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


# --------------------------------------------------------------------------
# FPS meter
# --------------------------------------------------------------------------
class FPSMeter:
    """Rolling-average FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque = deque(maxlen=max(2, window))
        self._last = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            self._times.append(now - self._last)
        self._last = now
        return self.fps

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        mean = float(np.mean(self._times))
        return 1.0 / mean if mean > 0 else 0.0


def print_banner(lines: Iterable[str]) -> None:
    lines = list(lines)
    width = max((len(line) for line in lines), default=0) + 4
    print("=" * width)
    for line in lines:
        print(f"  {line}")
    print("=" * width)
