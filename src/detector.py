"""
YOLOv8 detection layer.

Two independent detectors:
  * PersonDetector - plain COCO YOLOv8 (class 0 = person)
  * PPEDetector    - CUSTOM trained YOLOv8 weights (helmet / vest / gloves)

A COCO model can NOT detect PPE. If the custom weights are missing the PPE
detector stays disabled and the rest of the pipeline keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import Box, bottom_center, box_center, normalise_label, resolve_path

try:
    from ultralytics import YOLO

    _ULTRALYTICS_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - only hit on broken installs
    YOLO = None  # type: ignore[assignment]
    _ULTRALYTICS_ERROR = exc


@dataclass
class Detection:
    """A single YOLO detection."""

    bbox: Box                      # (x1, y1, x2, y2) in pixels
    confidence: float
    class_id: int
    label: str                     # raw model class name
    item: str = ""                 # canonical PPE item (helmet/vest/gloves) or "person"

    @property
    def center(self) -> Tuple[float, float]:
        return box_center(self.bbox)

    @property
    def foot_point(self) -> Tuple[float, float]:
        return bottom_center(self.bbox)

    def as_int_box(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        return int(x1), int(y1), int(x2), int(y2)


class _BaseYOLODetector:
    """Thin wrapper around ultralytics YOLO with friendly error handling."""

    def __init__(self, weights: str | Path, device: str = "cpu",
                 conf: float = 0.25, iou: float = 0.45,
                 imgsz: Optional[int] = 640) -> None:
        if YOLO is None:
            raise RuntimeError(
                "Ultralytics is not installed or failed to import "
                f"({_ULTRALYTICS_ERROR}).\nRun:  pip install -r requirements.txt"
            )
        self.weights = str(weights)
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz) if imgsz else 640
        self.model = YOLO(self.weights)
        try:
            self.model.to(device)
        except Exception as exc:
            print(f"[DETECTOR] Could not move model to '{device}' ({exc}); using CPU.")
            self.device = "cpu"
        # {class_id: class_name}
        names = getattr(self.model, "names", {}) or {}
        self.names: Dict[int, str] = {int(k): str(v) for k, v in names.items()}

    def _predict(self, frame: np.ndarray):
        return self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

    @staticmethod
    def _iter_boxes(results):
        if not results:
            return
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            yield (x1, y1, x2, y2), float(conf[i]), int(cls[i])


class PersonDetector(_BaseYOLODetector):
    """Detects people with a standard COCO YOLOv8 model."""

    PERSON_CLASS_NAMES = {"person", "people", "worker", "human"}

    def __init__(self, weights: str | Path, device: str = "cpu",
                 conf: float = 0.45, iou: float = 0.45,
                 imgsz: Optional[int] = 640,
                 min_box_area: int = 400) -> None:
        super().__init__(weights, device, conf, iou, imgsz)
        self.min_box_area = int(min_box_area)
        self.person_ids = {
            cid for cid, name in self.names.items()
            if normalise_label(name) in {normalise_label(n) for n in self.PERSON_CLASS_NAMES}
        }
        if not self.person_ids:
            # Custom person model with a single class -> accept everything.
            self.person_ids = set(self.names.keys())

    def detect(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        if frame is None or frame.size == 0:
            return detections
        results = self._predict(frame)
        for bbox, conf, cls_id in self._iter_boxes(results):
            if cls_id not in self.person_ids:
                continue
            x1, y1, x2, y2 = bbox
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                continue  # discard tiny/noisy boxes before they reach DeepSORT
            detections.append(
                Detection(bbox=bbox, confidence=conf, class_id=cls_id,
                          label=self.names.get(cls_id, "person"), item="person")
            )
        return detections


class PPEDetector(_BaseYOLODetector):
    """Detects helmet / vest / gloves with CUSTOM trained YOLOv8 weights."""

    def __init__(self, weights: str | Path, device: str = "cpu",
                 conf: float = 0.35, iou: float = 0.45,
                 imgsz: Optional[int] = 640,
                 class_aliases: Optional[Dict[str, List[str]]] = None,
                 ignore_classes: Optional[List[str]] = None) -> None:
        super().__init__(weights, device, conf, iou, imgsz)
        self.class_aliases = class_aliases or {}
        self.ignore = {normalise_label(c) for c in (ignore_classes or [])}
        self._alias_lookup: Dict[str, str] = {}
        for item, aliases in self.class_aliases.items():
            for alias in list(aliases) + [item]:
                self._alias_lookup[normalise_label(alias)] = item
        self._unmapped_warned: set = set()
        self.report_class_mapping()

    def report_class_mapping(self) -> None:
        print("[PPE MODEL] Classes found in weights:")
        for cid, name in sorted(self.names.items()):
            mapped = self._alias_lookup.get(normalise_label(name))
            if normalise_label(name) in self.ignore:
                note = "ignored"
            elif mapped:
                note = f"-> {mapped}"
            else:
                note = "-> (unmapped, add it to ppe.class_aliases in config.yaml)"
            print(f"           [{cid}] {name:<20} {note}")

    def canonical_item(self, label: str) -> Optional[str]:
        key = normalise_label(label)
        if key in self.ignore:
            return None
        item = self._alias_lookup.get(key)
        if item is None and key not in self._unmapped_warned:
            self._unmapped_warned.add(key)
        return item

    def detect(self, frame: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        if frame is None or frame.size == 0:
            return detections
        results = self._predict(frame)
        for bbox, conf, cls_id in self._iter_boxes(results):
            raw_label = self.names.get(cls_id, str(cls_id))
            item = self.canonical_item(raw_label)
            if item is None:
                continue
            detections.append(
                Detection(bbox=bbox, confidence=conf, class_id=cls_id,
                          label=raw_label, item=item)
            )
        return detections


class DisabledPPEDetector:
    """Stand-in used when the custom PPE weights are not present.

    Keeps the pipeline architecturally identical (and honest: it reports
    nothing rather than pretending COCO can see helmets).
    """

    available = False

    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    def detect(self, frame: np.ndarray) -> List[Detection]:  # noqa: ARG002
        return []


def build_ppe_detector(config: dict, device: str):
    """Create a PPEDetector, or a DisabledPPEDetector with a clear message."""
    ppe_path = resolve_path(config["models"]["ppe_model"])
    if not ppe_path.exists():
        msg = (
            "PPE model not found. Place your trained PPE YOLOv8 model at "
            "models/ppe_model/best.pt or update config.yaml."
        )
        print("\n" + "!" * 78)
        print(f"[PPE MODEL] {msg}")
        print(f"[PPE MODEL] Looked for: {ppe_path}")
        print("[PPE MODEL] Person detection, tracking, restricted zones and logging")
        print("[PPE MODEL] still run - PPE compliance will report every item as MISSING.")
        print("[PPE MODEL] The default COCO YOLOv8 model CANNOT detect helmet/vest/gloves.")
        print("!" * 78 + "\n")
        return DisabledPPEDetector(msg)

    det_cfg = config["detection"]
    ppe_cfg = config["ppe"]
    detector = PPEDetector(
        weights=ppe_path,
        device=device,
        conf=det_cfg["ppe_confidence"],
        iou=det_cfg["iou_threshold"],
        imgsz=det_cfg.get("imgsz", 640),
        class_aliases=ppe_cfg.get("class_aliases", {}),
        ignore_classes=ppe_cfg.get("ignore_classes", []),
    )
    setattr(detector, "available", True)
    return detector


def build_person_detector(config: dict, device: str) -> PersonDetector:
    person_path = config["models"]["person_model"]
    local = resolve_path(person_path)
    weights = str(local) if local.exists() else str(person_path)
    det_cfg = config["detection"]
    return PersonDetector(
        weights=weights,
        device=device,
        conf=det_cfg["person_confidence"],
        iou=det_cfg["iou_threshold"],
        imgsz=det_cfg.get("imgsz", 640),
    )
