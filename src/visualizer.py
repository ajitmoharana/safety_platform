"""
All OpenCV drawing: worker boxes, PPE boxes, restricted zones, alerts,
the live dashboard and the evidence-snapshot annotation.

Note on symbols: OpenCV's built-in Hershey fonts are ASCII only, so ticks and
crosses are drawn as [+] / [-] and the warning sign as "/!\\" instead of
unicode characters (which would render as '?').
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .compliance import PARTIAL, SAFE, UNSAFE, ComplianceResult
from .detector import Detection
from .logger import ViolationEvent
from .tracker import TrackedWorker
from .utils import clamp_box, seconds_to_hms, timestamp_now

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colours
COLOR_SAFE = (60, 200, 60)
COLOR_PARTIAL = (0, 215, 255)
COLOR_UNSAFE = (0, 0, 235)
COLOR_ZONE = (0, 0, 255)
COLOR_ZONE_ALERT = (0, 90, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (32, 32, 32)
COLOR_ACCENT = (0, 200, 255)

PPE_COLORS = {
    "helmet": (255, 190, 0),
    "vest": (0, 255, 200),
    "gloves": (255, 0, 200),
}

STATUS_COLORS = {SAFE: COLOR_SAFE, PARTIAL: COLOR_PARTIAL, UNSAFE: COLOR_UNSAFE}


def status_color(status: str) -> Tuple[int, int, int]:
    return STATUS_COLORS.get(status, COLOR_UNSAFE)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def draw_text(img: np.ndarray, text: str, org: Tuple[int, int],
              scale: float = 0.55, color=COLOR_TEXT, thickness: int = 1,
              shadow: bool = True) -> None:
    if shadow:
        cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_filled_rect(img: np.ndarray, pt1, pt2, color, alpha: float = 0.55) -> None:
    x1, y1 = (int(max(0, pt1[0])), int(max(0, pt1[1])))
    x2, y2 = (int(min(img.shape[1], pt2[0])), int(min(img.shape[0], pt2[1])))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full(roi.shape, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


class Visualizer:
    """Draws every overlay element on the processed frame."""

    def __init__(self, config: dict) -> None:
        disp = config.get("display", {})
        self.show_ppe_boxes = bool(disp.get("show_ppe_boxes", True))
        self.required = list(config.get("ppe", {}).get(
            "required", ["helmet", "vest", "gloves"]))

    # ------------------------------------------------------------------
    def draw_zones(self, frame: np.ndarray, zones: Sequence[np.ndarray],
                   alert_zones: Optional[set] = None) -> None:
        alert_zones = alert_zones or set()
        if not len(zones):
            return
        overlay = frame.copy()
        for idx, poly in enumerate(zones):
            colour = COLOR_ZONE_ALERT if idx in alert_zones else COLOR_ZONE
            cv2.fillPoly(overlay, [poly], colour)
        alpha = 0.35 if alert_zones else 0.22
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        for idx, poly in enumerate(zones):
            alert = idx in alert_zones
            colour = COLOR_ZONE_ALERT if alert else COLOR_ZONE
            cv2.polylines(frame, [poly], True, colour, 3 if alert else 2, cv2.LINE_AA)
            x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
            label = "!! RESTRICTED ZONE - INTRUSION" if alert else "RESTRICTED ZONE"
            draw_filled_rect(frame, (x, max(0, y - 24)),
                             (x + 12 + 9 * len(label), max(0, y - 24) + 22),
                             colour, 0.75)
            draw_text(frame, label, (x + 6, max(14, y - 8)), 0.55, COLOR_TEXT, 1, False)

    # ------------------------------------------------------------------
    def draw_ppe(self, frame: np.ndarray, ppe_detections: Sequence[Detection]) -> None:
        if not self.show_ppe_boxes:
            return
        for det in ppe_detections:
            x1, y1, x2, y2 = clamp_box(det.bbox, frame.shape[1], frame.shape[0])
            colour = PPE_COLORS.get(det.item, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
            label = f"{det.item} {det.confidence:.2f}"
            draw_text(frame, label, (x1, max(12, y1 - 4)), 0.42, colour, 1)

    # ------------------------------------------------------------------
    def draw_worker(self, frame: np.ndarray, worker: TrackedWorker,
                    result: Optional[ComplianceResult], in_zone: bool) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = clamp_box(worker.bbox, w, h)
        status = result.status if result else "UNKNOWN"
        colour = status_color(status) if result else (180, 180, 180)
        thickness = 3 if in_zone else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

        # ID banner
        title = worker.name
        draw_filled_rect(frame, (x1, max(0, y1 - 22)),
                         (x1 + 14 + 11 * len(title), max(0, y1 - 22) + 20),
                         colour, 0.85)
        draw_text(frame, title, (x1 + 6, max(14, y1 - 7)), 0.55, (0, 0, 0), 1, False)

        # PPE checklist panel under the ID
        lines: List[str] = []
        if result:
            for item in self.required:
                mark = "[+]" if result.items.get(item) else "[-]"
                lines.append(f"{item.capitalize():<7}{mark}")
            lines.append(f"STATUS: {status}")
        else:
            lines.append("PPE: n/a")

        panel_w = 118
        panel_h = 16 * len(lines) + 8
        py = min(h - panel_h - 2, y1 + 2)
        draw_filled_rect(frame, (x1, py), (x1 + panel_w, py + panel_h), COLOR_PANEL, 0.62)
        ty = py + 16
        for i, line in enumerate(lines):
            line_colour = colour if i == len(lines) - 1 else COLOR_TEXT
            draw_text(frame, line, (x1 + 6, ty), 0.45, line_colour, 1, False)
            ty += 16

        # ground reference point
        fx, fy = worker.foot_point
        cv2.circle(frame, (int(fx), int(fy)), 4, colour, -1)

        if in_zone:
            warn = f"/!\\ RESTRICTED ZONE - {worker.name}"
            wy = min(h - 8, y2 + 22)
            draw_filled_rect(frame, (x1, wy - 18), (x1 + 12 + 10 * len(warn), wy + 6),
                             COLOR_UNSAFE, 0.85)
            draw_text(frame, warn, (x1 + 6, wy), 0.5, COLOR_TEXT, 1, False)

    # ------------------------------------------------------------------
    def draw_dashboard(self, frame: np.ndarray, stats: Dict[str, float],
                       fps: float, video_time: float, device: str,
                       ppe_model_available: bool = True) -> None:
        h, w = frame.shape[:2]
        panel_w, panel_h = 256, 248 if not ppe_model_available else 230
        x0, y0 = w - panel_w - 12, 12
        draw_filled_rect(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), COLOR_PANEL, 0.72)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), COLOR_ACCENT, 1)

        draw_text(frame, "SITE SAFETY MONITOR", (x0 + 12, y0 + 24), 0.58,
                  COLOR_ACCENT, 2, False)
        cv2.line(frame, (x0 + 10, y0 + 32), (x0 + panel_w - 10, y0 + 32),
                 COLOR_ACCENT, 1)

        rows = [
            (f"Workers: {int(stats.get('workers', 0))}", COLOR_TEXT),
            (f"Safe: {int(stats.get('safe', 0))}", COLOR_SAFE),
            (f"Partial: {int(stats.get('partial', 0))}", COLOR_PARTIAL),
            (f"Unsafe: {int(stats.get('unsafe', 0))}", COLOR_UNSAFE),
            (f"PPE Compliance: {stats.get('compliance', 0.0):.0f}%", COLOR_TEXT),
            (f"Violations: {int(stats.get('violations', 0))}", COLOR_TEXT),
            (f"Restricted Entries: {int(stats.get('zone_entries', 0))}", COLOR_TEXT),
        ]
        y = y0 + 54
        for text, colour in rows:
            draw_text(frame, text, (x0 + 12, y), 0.5, colour, 1, False)
            y += 20

        cv2.line(frame, (x0 + 10, y - 12), (x0 + panel_w - 10, y - 12), (90, 90, 90), 1)
        draw_text(frame, f"FPS: {fps:4.1f}   Device: {device}",
                  (x0 + 12, y + 4), 0.45, COLOR_TEXT, 1, False)
        draw_text(frame, f"Video: {seconds_to_hms(video_time)}   {timestamp_now('%H:%M:%S')}",
                  (x0 + 12, y + 22), 0.45, COLOR_TEXT, 1, False)

        if not ppe_model_available:
            draw_text(frame, "PPE MODEL MISSING", (x0 + 12, y + 40), 0.45,
                      COLOR_UNSAFE, 1, False)

    # ------------------------------------------------------------------
    def draw_alerts(self, frame: np.ndarray, zone_workers: Sequence[int]) -> None:
        if not zone_workers:
            return
        h, w = frame.shape[:2]
        ids = ", ".join(f"#{i}" for i in sorted(zone_workers))
        banner = f"/!\\ VIOLATION: WORKER {ids} ENTERED RESTRICTED ZONE"
        draw_filled_rect(frame, (0, 0), (w, 42), COLOR_UNSAFE, 0.70)
        draw_text(frame, banner, (16, 29), 0.72, COLOR_TEXT, 2, False)

    def draw_paused(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        draw_text(frame, "PAUSED - press P to resume", (16, h - 18), 0.7,
                  COLOR_PARTIAL, 2)

    def draw_help(self, frame: np.ndarray, seconds_left: float) -> None:
        if seconds_left <= 0:
            return
        h, w = frame.shape[:2]
        lines = ["CONTROLS", "Q = Quit", "P = Pause / Resume",
                 "R = Reset restricted zone", "S = Save screenshot"]
        x0, y0 = 12, h - (22 * len(lines)) - 20
        draw_filled_rect(frame, (x0, y0), (x0 + 230, y0 + 22 * len(lines) + 12),
                         COLOR_PANEL, 0.65)
        y = y0 + 22
        for i, line in enumerate(lines):
            draw_text(frame, line, (x0 + 10, y), 0.5,
                      COLOR_ACCENT if i == 0 else COLOR_TEXT, 1, False)
            y += 22

    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray,
               workers: Sequence[TrackedWorker],
               compliance: Dict[int, ComplianceResult],
               ppe_detections: Sequence[Detection],
               zones: Sequence[np.ndarray],
               zone_hits: Dict[int, int],
               stats: Dict[str, float],
               fps: float,
               video_time: float,
               device: str,
               ppe_model_available: bool = True,
               help_seconds_left: float = 0.0,
               paused: bool = False) -> np.ndarray:
        """Draw the full overlay stack and return the annotated frame."""
        out = frame  # drawn in-place on the caller's copy
        self.draw_zones(out, zones, set(zone_hits.values()))
        self.draw_ppe(out, ppe_detections)
        for worker in workers:
            self.draw_worker(out, worker, compliance.get(worker.track_id),
                             worker.track_id in zone_hits)
        self.draw_alerts(out, list(zone_hits.keys()))
        self.draw_dashboard(out, stats, fps, video_time, device, ppe_model_available)
        self.draw_help(out, help_seconds_left)
        if paused:
            self.draw_paused(out)
        return out

    # ------------------------------------------------------------------
    def make_evidence_frame(self, annotated: np.ndarray,
                            event: ViolationEvent) -> np.ndarray:
        """Highlight the offending worker and stamp the violation banner."""
        evidence = annotated.copy()
        h, w = evidence.shape[:2]
        x1, y1, x2, y2 = clamp_box(event.bbox, w, h)
        cv2.rectangle(evidence, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 3)

        banner = (f"EVIDENCE | Worker #{event.worker_id} | {event.violation_type} | "
                  f"{event.ppe_status}")
        detail = (f"Helmet:{event.helmet}  Vest:{event.vest}  Gloves:{event.gloves}  "
                  f"Zone:{event.restricted_zone}  Time:{event.timestamp} "
                  f"({event.video_time})")
        draw_filled_rect(evidence, (0, h - 56), (w, h), (0, 0, 0), 0.72)
        draw_text(evidence, banner, (14, h - 32), 0.62, (0, 0, 255), 2, False)
        draw_text(evidence, detail, (14, h - 12), 0.48, COLOR_TEXT, 1, False)
        return evidence
