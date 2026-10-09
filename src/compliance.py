"""
PPE compliance classification with temporal smoothing.

A single missed YOLO frame must not flip a worker SAFE -> UNSAFE -> SAFE.
For every tracked worker we keep a short history (default 15 frames) of raw
per-item detections and use majority-style voting: an item counts as WORN when
it was detected in at least `presence_ratio` of the recent history.

Statuses:
  SAFE    - every required item present
  PARTIAL - some present, some missing
  UNSAFE  - nothing present, or >= `unsafe_if_missing_at_least` items missing
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .ppe_matcher import WorkerPPE

SAFE = "SAFE"
PARTIAL = "PARTIAL"
UNSAFE = "UNSAFE"


@dataclass
class ComplianceResult:
    """Smoothed PPE state for one worker in the current frame."""

    worker_id: int
    items: Dict[str, bool] = field(default_factory=dict)
    status: str = UNSAFE
    confidence: float = 0.0
    frames_tracked: int = 0

    @property
    def missing(self) -> List[str]:
        return [item for item, worn in self.items.items() if not worn]

    @property
    def present(self) -> List[str]:
        return [item for item, worn in self.items.items() if worn]

    def flag(self, item: str) -> str:
        return "OK" if self.items.get(item, False) else "X"

    def summary_lines(self) -> List[str]:
        lines = [f"{item.capitalize()}: {'[+]' if worn else '[-]'}"
                 for item, worn in self.items.items()]
        lines.append(f"Status: {self.status}")
        return lines


class ComplianceAnalyzer:
    """Maintains per-worker PPE history and classifies compliance."""

    def __init__(self, config: dict) -> None:
        comp = config.get("compliance", {})
        self.required: List[str] = list(config.get("ppe", {}).get(
            "required", ["helmet", "vest", "gloves"]))
        self.window = max(1, int(comp.get("smoothing_frames", 15)))
        self.presence_ratio = float(comp.get("presence_ratio", 0.35))
        self.min_history = max(1, int(comp.get("min_history", 5)))
        self.unsafe_if_missing_at_least = int(comp.get("unsafe_if_missing_at_least", 2))

        self._history: Dict[int, Dict[str, deque]] = {}
        self._conf_history: Dict[int, deque] = {}
        self._frames_seen: Dict[int, int] = {}
        self._last_seen_frame: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def _ensure(self, worker_id: int) -> None:
        if worker_id not in self._history:
            self._history[worker_id] = {
                item: deque(maxlen=self.window) for item in self.required
            }
            self._conf_history[worker_id] = deque(maxlen=self.window)
            self._frames_seen[worker_id] = 0

    def update(self, worker_ppe: WorkerPPE, frame_index: int = 0) -> ComplianceResult:
        """Push one raw observation and return the smoothed result."""
        wid = worker_ppe.worker_id
        self._ensure(wid)
        self._frames_seen[wid] += 1
        self._last_seen_frame[wid] = frame_index
        self._conf_history[wid].append(worker_ppe.mean_confidence)

        smoothed: Dict[str, bool] = {}
        for item in self.required:
            hist = self._history[wid][item]
            hist.append(1 if worker_ppe.has(item) else 0)
            if len(hist) < self.min_history:
                # Not enough evidence yet -> trust the latest raw observation.
                smoothed[item] = bool(hist[-1])
            else:
                smoothed[item] = (sum(hist) / len(hist)) >= self.presence_ratio

        status = self.classify(smoothed)
        conf_values = [c for c in self._conf_history[wid] if c > 0]
        confidence = float(sum(conf_values) / len(conf_values)) if conf_values else 0.0
        return ComplianceResult(worker_id=wid, items=smoothed, status=status,
                                confidence=round(confidence, 3),
                                frames_tracked=self._frames_seen[wid])

    # ------------------------------------------------------------------
    def classify(self, items: Dict[str, bool]) -> str:
        """Apply the configurable SAFE / PARTIAL / UNSAFE rules."""
        required = [i for i in self.required if i in items]
        if not required:
            return UNSAFE
        present = sum(1 for i in required if items[i])
        missing = len(required) - present
        if missing == 0:
            return SAFE
        if present == 0 or missing >= self.unsafe_if_missing_at_least:
            return UNSAFE
        return PARTIAL

    # ------------------------------------------------------------------
    def analyse_all(self, matched: Dict[int, WorkerPPE],
                    frame_index: int = 0) -> Dict[int, ComplianceResult]:
        return {wid: self.update(wp, frame_index) for wid, wp in matched.items()}

    def cleanup(self, frame_index: int, max_absent_frames: int = 150) -> None:
        """Drop history for workers that disappeared long ago (memory hygiene)."""
        stale = [wid for wid, last in self._last_seen_frame.items()
                 if frame_index - last > max_absent_frames]
        for wid in stale:
            self._history.pop(wid, None)
            self._conf_history.pop(wid, None)
            self._frames_seen.pop(wid, None)
            self._last_seen_frame.pop(wid, None)


def summarise(results: Sequence[ComplianceResult]) -> Dict[str, float]:
    """Aggregate statistics used by the live dashboard."""
    total = len(results)
    safe = sum(1 for r in results if r.status == SAFE)
    partial = sum(1 for r in results if r.status == PARTIAL)
    unsafe = sum(1 for r in results if r.status == UNSAFE)
    compliance = (safe / total * 100.0) if total else 0.0
    return {"workers": total, "safe": safe, "partial": partial,
            "unsafe": unsafe, "compliance": compliance}
