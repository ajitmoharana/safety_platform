"""
Violation logging (CSV + JSON) and evidence snapshot writing.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .utils import file_timestamp, resolve_path, safe_filename, timestamp_now

CSV_COLUMNS = [
    "event_id",
    "timestamp",
    "video_time",
    "worker_id",
    "violation_type",
    "helmet",
    "vest",
    "gloves",
    "ppe_status",
    "restricted_zone",
    "confidence",
    "snapshot_path",
    "source",
]


@dataclass
class ViolationEvent:
    """One confirmed violation, ready to be written to disk."""

    event_id: str = ""
    timestamp: str = ""
    video_time: str = "00:00:00"
    worker_id: int = 0
    violation_type: str = ""
    helmet: str = "NO"
    vest: str = "NO"
    gloves: str = "NO"
    ppe_status: str = "UNSAFE"
    restricted_zone: str = "NO"
    confidence: float = 0.0
    snapshot_path: str = ""
    source: str = ""
    # runtime-only fields (not written to CSV)
    bbox: tuple = field(default=(0, 0, 0, 0), repr=False)
    frame_index: int = field(default=0, repr=False)

    def to_row(self) -> Dict[str, Any]:
        return {col: getattr(self, col) for col in CSV_COLUMNS}


class ViolationLogger:
    """Appends violations to output/logs/violations.csv and saves snapshots."""

    def __init__(self, config: dict, source_name: str = "") -> None:
        out = config.get("output", {})
        self.save_logs = bool(out.get("save_logs", True))
        self.save_snapshots = bool(out.get("save_snapshots", True))
        self.append = bool(out.get("append_logs", True))
        self.csv_path = resolve_path(out.get("log_file", "output/logs/violations.csv"))
        self.snapshot_dir = resolve_path(out.get("snapshots_dir", "output/snapshots"))
        self.json_path = self.csv_path.with_suffix(".json")
        self.source_name = source_name
        self._counter = 0
        self._events: List[ViolationEvent] = []
        self._prepare()

    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"[LOG] Could not create output folders: {exc}")
            self.save_logs = False
            self.save_snapshots = False
            return

        if not self.append and self.csv_path.exists():
            try:
                self.csv_path.unlink()
            except OSError as exc:
                print(f"[LOG] Could not reset log file: {exc}")

        if self.csv_path.exists():
            self._counter = self._count_existing_rows()
        else:
            self._write_header()

    def _count_existing_rows(self) -> int:
        try:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as fh:
                return max(0, sum(1 for _ in fh) - 1)
        except OSError:
            return 0

    def _write_header(self) -> None:
        if not self.save_logs:
            return
        try:
            with open(self.csv_path, "w", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()
        except OSError as exc:
            print(f"[LOG] Could not create {self.csv_path}: {exc}")
            self.save_logs = False

    # ------------------------------------------------------------------
    def next_event_id(self) -> str:
        self._counter += 1
        return f"{self._counter:05d}"

    def save_snapshot(self, frame: np.ndarray, worker_id: int,
                      violation_type: str) -> str:
        """Write the ANNOTATED evidence frame; return its relative path."""
        if not self.save_snapshots or frame is None or frame.size == 0:
            return ""
        name = (f"worker_{worker_id}_{safe_filename(violation_type.lower())}"
                f"_{file_timestamp()}.jpg")
        path = self.snapshot_dir / name
        try:
            ok = cv2.imwrite(str(path), frame)
            if not ok:
                print(f"[LOG] Failed to write snapshot {path}")
                return ""
        except (cv2.error, OSError) as exc:
            print(f"[LOG] Snapshot error: {exc}")
            return ""
        try:
            return str(path.relative_to(resolve_path(".")))
        except ValueError:
            return str(path)

    def log(self, event: ViolationEvent) -> None:
        """Append one event to the CSV (and keep it for the JSON summary)."""
        self._events.append(event)
        if not self.save_logs:
            return
        try:
            with open(self.csv_path, "a", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writerow(event.to_row())
        except OSError as exc:
            print(f"[LOG] Could not append to {self.csv_path}: {exc}")

    def export_json(self) -> None:
        """Write a JSON copy of this run's events (handy for dashboards)."""
        if not self.save_logs or not self._events:
            return
        try:
            payload = {
                "source": self.source_name,
                "generated_at": timestamp_now(),
                "total_events": len(self._events),
                "events": [e.to_row() for e in self._events],
            }
            with open(self.json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as exc:
            print(f"[LOG] Could not write JSON log: {exc}")

    # ------------------------------------------------------------------
    @property
    def events(self) -> List[ViolationEvent]:
        return list(self._events)

    @property
    def total(self) -> int:
        return len(self._events)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ev in self._events:
            counts[ev.violation_type] = counts.get(ev.violation_type, 0) + 1
        return counts
