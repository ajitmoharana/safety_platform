"""
DeepSORT worker tracking.

Wraps `deep_sort_realtime` so the rest of the project only ever sees a simple
list of TrackedWorker objects with stable integer IDs ("Worker #12").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .detector import Detection
from .utils import Box, bottom_center, box_center, clamp_box, xyxy_to_ltwh

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort

    _DEEPSORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    DeepSort = None  # type: ignore[assignment]
    _DEEPSORT_ERROR = exc


@dataclass
class TrackedWorker:
    """One confirmed DeepSORT track."""

    track_id: int
    bbox: Box
    confidence: float = 0.0
    age: int = 0

    @property
    def name(self) -> str:
        return f"Worker #{self.track_id}"

    @property
    def center(self) -> Tuple[float, float]:
        return box_center(self.bbox)

    @property
    def foot_point(self) -> Tuple[float, float]:
        return bottom_center(self.bbox)

    def reference_point(self, mode: str = "bottom_center") -> Tuple[float, float]:
        return self.foot_point if mode == "bottom_center" else self.center


class WorkerTracker:
    """DeepSORT tracker for person detections."""

    def __init__(self, config: dict, device: str = "cpu") -> None:
        if DeepSort is None:
            raise RuntimeError(
                "deep-sort-realtime is not installed or failed to import "
                f"({_DEEPSORT_ERROR}).\nRun:  pip install deep-sort-realtime"
            )
        cfg = config.get("tracking", {})
        embedder = cfg.get("embedder", "mobilenet")
        half = bool(cfg.get("half", False)) and str(device).startswith("cuda")
        try:
            self.tracker = DeepSort(
                max_age=int(cfg.get("max_age", 30)),
                n_init=int(cfg.get("n_init", 3)),
                max_iou_distance=float(cfg.get("max_iou_distance", 0.7)),
                max_cosine_distance=float(cfg.get("max_cosine_distance", 0.3)),
                embedder=embedder,
                half=half,
                bgr=True,
                embedder_gpu=str(device).startswith("cuda"),
            )
        except Exception as exc:
            # e.g. embedder weights could not be downloaded -> retry on CPU basics
            print(f"[TRACKER] DeepSORT init failed with embedder='{embedder}' ({exc}).")
            print("[TRACKER] Retrying with CPU embedder settings...")
            self.tracker = DeepSort(
                max_age=int(cfg.get("max_age", 30)),
                n_init=int(cfg.get("n_init", 3)),
                max_iou_distance=float(cfg.get("max_iou_distance", 0.7)),
                embedder="mobilenet",
                half=False,
                bgr=True,
                embedder_gpu=False,
            )

    @staticmethod
    def _to_raw_detections(detections: Sequence[Detection]) -> List[tuple]:
        """DeepSORT expects ([left, top, w, h], confidence, class)."""
        raw = []
        for det in detections:
            raw.append((xyxy_to_ltwh(det.bbox), float(det.confidence), "person"))
        return raw

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> List[TrackedWorker]:
        """Feed YOLO person detections to DeepSORT; return confirmed tracks."""
        height, width = frame.shape[:2]
        raw = self._to_raw_detections(detections)
        try:
            tracks = self.tracker.update_tracks(raw, frame=frame)
        except Exception as exc:
            print(f"[TRACKER] update_tracks failed on this frame: {exc}")
            return []

        workers: List[TrackedWorker] = []
        det_boxes = [d.bbox for d in detections]
        det_confs = [d.confidence for d in detections]
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            try:
                x1, y1, x2, y2 = track.to_ltrb()
            except Exception:
                continue
            bbox = clamp_box((x1, y1, x2, y2), width, height)
            if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
                continue
            conf = self._nearest_confidence(bbox, det_boxes, det_confs)
            try:
                track_id = int(track.track_id)
            except (TypeError, ValueError):
                track_id = abs(hash(track.track_id)) % 100000
            workers.append(
                TrackedWorker(track_id=track_id, bbox=bbox, confidence=conf,
                              age=int(getattr(track, "age", 0) or 0))
            )
        return workers

    @staticmethod
    def _nearest_confidence(bbox: Box, det_boxes: List[Box], det_confs: List[float]) -> float:
        """Attach the detection confidence of the best-overlapping YOLO box."""
        from .utils import iou as _iou  # local import keeps module import order simple

        best, best_iou = 0.0, 0.0
        for box, conf in zip(det_boxes, det_confs):
            score = _iou(bbox, box)
            if score > best_iou:
                best_iou, best = score, conf
        return float(best)
