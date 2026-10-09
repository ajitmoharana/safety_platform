"""
PPE -> Person matching.

PPE is never counted globally: every helmet / vest / glove box is assigned to
exactly one tracked worker, using

  * containment of the PPE box inside the worker box,
  * the expected vertical band for that item (helmet on top, vest on the
    torso, gloves lower / around the body),
  * detection confidence as a tie-breaker.

Each PPE detection is given to its single best-scoring worker (greedy
assignment on the sorted score list), so two workers can never "share" the
same helmet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .detector import Detection
from .tracker import TrackedWorker
from .utils import box_center, containment


@dataclass
class WorkerPPE:
    """Raw (per-frame) PPE result for one worker."""

    worker_id: int
    items: Dict[str, bool] = field(default_factory=dict)
    detections: Dict[str, Detection] = field(default_factory=dict)

    def has(self, item: str) -> bool:
        return bool(self.items.get(item, False))

    def confidence(self, item: str) -> float:
        det = self.detections.get(item)
        return float(det.confidence) if det else 0.0

    @property
    def mean_confidence(self) -> float:
        confs = [d.confidence for d in self.detections.values()]
        return float(sum(confs) / len(confs)) if confs else 0.0


class PPEMatcher:
    """Assigns detected PPE boxes to tracked workers."""

    def __init__(self, config: dict) -> None:
        ppe_cfg = config.get("ppe", {})
        match_cfg = config.get("matching", {})
        self.required: List[str] = list(ppe_cfg.get("required", ["helmet", "vest", "gloves"]))
        self.min_containment = float(match_cfg.get("min_containment", 0.55))
        self.regions: Dict[str, Sequence[float]] = dict(match_cfg.get("regions", {}))
        self.out_of_region_penalty = float(match_cfg.get("out_of_region_penalty", 0.35))

    # ------------------------------------------------------------------
    def _region_score(self, item: str, ppe_box, worker_box) -> float:
        """1.0 when the PPE sits in its expected vertical band, else penalised."""
        band = self.regions.get(item)
        if not band:
            return 1.0
        wy1, wy2 = worker_box[1], worker_box[3]
        height = max(1e-6, wy2 - wy1)
        _, cy = box_center(ppe_box)
        rel = (cy - wy1) / height
        low, high = float(band[0]), float(band[1])
        if low <= rel <= high:
            return 1.0
        # Smoothly decay outside the band instead of rejecting outright.
        distance = (low - rel) if rel < low else (rel - high)
        return max(0.0, 1.0 - self.out_of_region_penalty - distance)

    def _score(self, item: str, ppe: Detection, worker: TrackedWorker) -> float:
        cov = containment(ppe.bbox, worker.bbox)
        if cov < self.min_containment:
            return 0.0
        region = self._region_score(item, ppe.bbox, worker.bbox)
        if region <= 0.0:
            return 0.0
        # weighted blend: overlap dominates, region + confidence refine it
        return 0.6 * cov + 0.3 * region + 0.1 * float(ppe.confidence)

    # ------------------------------------------------------------------
    def match(self, workers: Sequence[TrackedWorker],
              ppe_detections: Sequence[Detection]) -> Dict[int, WorkerPPE]:
        """Return {worker_id: WorkerPPE} for every tracked worker."""
        result: Dict[int, WorkerPPE] = {
            w.track_id: WorkerPPE(worker_id=w.track_id,
                                  items={item: False for item in self.required})
            for w in workers
        }
        if not workers or not ppe_detections:
            return result

        # Build every (score, ppe_index, worker) candidate pair.
        candidates: List[Tuple[float, int, TrackedWorker]] = []
        for idx, ppe in enumerate(ppe_detections):
            item = ppe.item
            if item not in self.required:
                continue
            for worker in workers:
                score = self._score(item, ppe, worker)
                if score > 0:
                    candidates.append((score, idx, worker))

        candidates.sort(key=lambda c: c[0], reverse=True)

        used_ppe: set = set()
        for score, idx, worker in candidates:
            if idx in used_ppe:
                continue  # this PPE box already belongs to someone
            ppe = ppe_detections[idx]
            entry = result[worker.track_id]
            existing = entry.detections.get(ppe.item)
            if existing is not None and existing.confidence >= ppe.confidence:
                continue  # worker already has a better box for this item
            entry.items[ppe.item] = True
            entry.detections[ppe.item] = ppe
            used_ppe.add(idx)

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def unmatched(ppe_detections: Sequence[Detection],
                  matched: Dict[int, WorkerPPE]) -> List[Detection]:
        """PPE boxes that could not be attributed to any worker."""
        assigned = {id(d) for wp in matched.values() for d in wp.detections.values()}
        return [d for d in ppe_detections if id(d) not in assigned]
