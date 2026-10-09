"""
Restricted zone definition and intrusion detection.

Zones are polygons. They can be:
  * drawn interactively with the mouse on the first video frame,
  * loaded from data/zones.json (saved from a previous run),
  * or read from config.yaml -> restricted_zone.polygons.

Intrusion uses cv2.pointPolygonTest() on the worker's ground reference point
(bottom-centre of the DeepSORT bounding box).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .tracker import TrackedWorker
from .utils import resolve_path


class RestrictedZoneManager:
    """Holds polygons and answers "is this worker inside a zone?"."""

    def __init__(self, config: dict) -> None:
        cfg = config.get("restricted_zone", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.interactive = bool(cfg.get("interactive_selection", True))
        self.zone_file = resolve_path(cfg.get("zone_file", "data/zones.json"))
        self.load_saved = bool(cfg.get("load_saved_zones", True))
        self.reference_mode = str(cfg.get("reference_point", "bottom_center"))
        self.zones: List[np.ndarray] = []
        self._config_polygons = cfg.get("polygons", []) or []

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------
    def load_from_config(self) -> bool:
        zones = []
        for poly in self._config_polygons:
            pts = self._to_array(poly)
            if pts is not None:
                zones.append(pts)
        if zones:
            self.zones = zones
            print(f"[ZONE] Loaded {len(zones)} zone(s) from config.yaml.")
            return True
        return False

    def load_from_file(self) -> bool:
        if not self.zone_file.exists():
            return False
        try:
            with open(self.zone_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            polygons = data.get("zones", data) if isinstance(data, dict) else data
            zones = []
            for poly in polygons:
                pts = self._to_array(poly)
                if pts is not None:
                    zones.append(pts)
            if zones:
                self.zones = zones
                print(f"[ZONE] Loaded {len(zones)} saved zone(s) from {self.zone_file}.")
                return True
        except (OSError, ValueError, TypeError) as exc:
            print(f"[ZONE] Could not read {self.zone_file}: {exc}")
        return False

    def save(self) -> None:
        try:
            self.zone_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"zones": [z.tolist() for z in self.zones]}
            with open(self.zone_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"[ZONE] Saved {len(self.zones)} zone(s) to {self.zone_file}.")
        except OSError as exc:
            print(f"[ZONE] Could not save zones: {exc}")

    @staticmethod
    def _to_array(points: Sequence) -> Optional[np.ndarray]:
        try:
            arr = np.array(points, dtype=np.int32).reshape(-1, 2)
        except (ValueError, TypeError):
            return None
        return arr if len(arr) >= 3 else None

    # ------------------------------------------------------------------
    # Interactive selection
    # ------------------------------------------------------------------
    def select_interactively(self, frame: np.ndarray,
                             window_name: str = "Define Restricted Zone") -> bool:
        """Let the user click polygon points on a still frame.

        Controls: Left click = add point | Right click / ENTER = close polygon
                  R = reset | S = save & continue | ESC = cancel
        Returns True when at least one zone exists afterwards.
        """
        if frame is None or frame.size == 0:
            print("[ZONE] No frame available for interactive selection.")
            return bool(self.zones)

        current: List[Tuple[int, int]] = []
        polygons: List[np.ndarray] = list(self.zones)

        def on_mouse(event, x, y, flags, param):  # noqa: ANN001, ARG001
            nonlocal current, polygons
            if event == cv2.EVENT_LBUTTONDOWN:
                current.append((int(x), int(y)))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(current) >= 3:
                    polygons.append(np.array(current, dtype=np.int32))
                    current = []
                else:
                    print("[ZONE] A polygon needs at least 3 points.")

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, on_mouse)
        except cv2.error as exc:
            print(f"[ZONE] Interactive selection unavailable (no GUI?): {exc}")
            return bool(self.zones)

        help_lines = [
            "DEFINE RESTRICTED ZONE",
            "Left click  : add point",
            "Right click : close polygon",
            "ENTER       : close polygon / finish",
            "R           : reset all",
            "S           : save and start monitoring",
            "ESC         : skip (no zone)",
        ]

        while True:
            canvas = frame.copy()
            overlay = canvas.copy()
            for poly in polygons:
                cv2.fillPoly(overlay, [poly], (0, 0, 200))
            cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)
            for poly in polygons:
                cv2.polylines(canvas, [poly], True, (0, 0, 255), 2)
            if current:
                pts = np.array(current, dtype=np.int32)
                cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)
                for pt in current:
                    cv2.circle(canvas, pt, 4, (0, 255, 255), -1)

            y = 28
            for i, line in enumerate(help_lines):
                scale = 0.7 if i == 0 else 0.55
                cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (255, 255, 255), 1, cv2.LINE_AA)
                y += 26
            cv2.putText(canvas, f"Zones: {len(polygons)}  Points: {len(current)}",
                        (14, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        cv2.LINE_AA)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):            # ENTER
                if len(current) >= 3:
                    polygons.append(np.array(current, dtype=np.int32))
                    current = []
                else:
                    break                   # ENTER with no pending points = finish
            elif key in (ord("r"), ord("R")):
                current, polygons = [], []
            elif key in (ord("s"), ord("S")):
                if len(current) >= 3:
                    polygons.append(np.array(current, dtype=np.int32))
                    current = []
                break
            elif key == 27:                 # ESC
                polygons = list(self.zones)
                break

        cv2.destroyWindow(window_name)
        self.zones = polygons
        if self.zones:
            self.save()
        else:
            print("[ZONE] No restricted zone defined - zone monitoring inactive.")
        return bool(self.zones)

    def setup(self, first_frame: np.ndarray, force_interactive: bool = False) -> None:
        """Resolve zones from file/config/mouse depending on configuration."""
        if not self.enabled:
            print("[ZONE] Restricted zone monitoring disabled in config.yaml.")
            return
        if force_interactive:
            self.select_interactively(first_frame)
            return
        if self.load_saved and self.load_from_file():
            return
        if self.load_from_config():
            return
        if self.interactive:
            self.select_interactively(first_frame)
        else:
            print("[ZONE] No zones configured and interactive selection is off.")

    # ------------------------------------------------------------------
    # Intrusion test
    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.enabled and len(self.zones) > 0

    def point_inside(self, point: Sequence[float]) -> int:
        """Return the index of the zone containing the point, else -1."""
        px, py = float(point[0]), float(point[1])
        for idx, poly in enumerate(self.zones):
            if cv2.pointPolygonTest(poly, (px, py), False) >= 0:
                return idx
        return -1

    def check_workers(self, workers: Sequence[TrackedWorker]) -> Dict[int, int]:
        """{worker_id: zone_index} for every worker standing inside a zone."""
        inside: Dict[int, int] = {}
        if not self.active:
            return inside
        for worker in workers:
            ref = worker.reference_point(self.reference_mode)
            idx = self.point_inside(ref)
            if idx >= 0:
                inside[worker.track_id] = idx
        return inside

    def reset(self) -> None:
        self.zones = []
