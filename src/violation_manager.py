"""
Violation detection with debouncing, persistence and cooldown.

Supported violation types
  MISSING_HELMET / MISSING_VEST / MISSING_GLOVES
  PPE_PARTIAL / PPE_UNSAFE
  RESTRICTED_ZONE_ENTRY

Rules that stop the log from filling up with one row per frame:
  * a violation must persist for `min_violation_frames` consecutive frames,
  * after logging, the same (worker, violation) pair is muted for
    `cooldown_seconds`,
  * when the worker becomes compliant the state resets, so a NEW violation
    later is logged immediately (no waiting for the cooldown).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .compliance import PARTIAL, UNSAFE, ComplianceResult
from .logger import ViolationEvent, ViolationLogger
from .tracker import TrackedWorker
from .utils import seconds_to_hms, timestamp_now

MISSING_HELMET = "MISSING_HELMET"
MISSING_VEST = "MISSING_VEST"
MISSING_GLOVES = "MISSING_GLOVES"
PPE_PARTIAL = "PPE_PARTIAL"
PPE_UNSAFE = "PPE_UNSAFE"
RESTRICTED_ZONE_ENTRY = "RESTRICTED_ZONE_ENTRY"

MISSING_TYPE_BY_ITEM = {
    "helmet": MISSING_HELMET,
    "vest": MISSING_VEST,
    "gloves": MISSING_GLOVES,
}


@dataclass
class _State:
    """Debounce state for one (worker, violation_type) pair."""

    active_frames: int = 0
    last_logged_time: float = -1e9
    was_violating: bool = False


class ViolationManager:
    """Decides which violations are real and hands them to the logger."""

    def __init__(self, config: dict, logger: ViolationLogger, source: str = "") -> None:
        vcfg = config.get("violations", {})
        self.cooldown = float(vcfg.get("cooldown_seconds", 8))
        self.min_frames = max(1, int(vcfg.get("min_violation_frames", 5)))
        self.save_snapshot = bool(vcfg.get("save_snapshot", True))
        self.enabled_types = {
            MISSING_HELMET: bool(vcfg.get("types", {}).get("missing_helmet", True)),
            MISSING_VEST: bool(vcfg.get("types", {}).get("missing_vest", True)),
            MISSING_GLOVES: bool(vcfg.get("types", {}).get("missing_gloves", True)),
            PPE_PARTIAL: bool(vcfg.get("types", {}).get("ppe_partial", True)),
            PPE_UNSAFE: bool(vcfg.get("types", {}).get("ppe_unsafe", True)),
            RESTRICTED_ZONE_ENTRY: bool(vcfg.get("types", {}).get(
                "restricted_zone_entry", True)),
        }
        self.logger = logger
        self.source = source
        self._states: Dict[Tuple[int, str], _State] = {}
        self.zone_entry_count = 0

    # ------------------------------------------------------------------
    def _state(self, worker_id: int, vtype: str) -> _State:
        key = (worker_id, vtype)
        if key not in self._states:
            self._states[key] = _State()
        return self._states[key]

    def _should_fire(self, worker_id: int, vtype: str, violating: bool,
                     now: float) -> bool:
        """Debounce + cooldown decision for one violation candidate."""
        state = self._state(worker_id, vtype)
        if not violating:
            state.active_frames = 0
            if state.was_violating:
                # worker became compliant -> allow an immediate re-log later
                state.was_violating = False
                state.last_logged_time = -1e9
            return False

        state.active_frames += 1
        if state.active_frames < self.min_frames:
            return False
        if (now - state.last_logged_time) < self.cooldown:
            return False
        state.last_logged_time = now
        state.was_violating = True
        return True

    # ------------------------------------------------------------------
    def evaluate(self,
                 workers: Sequence[TrackedWorker],
                 compliance: Dict[int, ComplianceResult],
                 zone_hits: Dict[int, int],
                 now: float,
                 video_time_seconds: float,
                 frame_index: int,
                 ppe_model_available: bool = True) -> List[ViolationEvent]:
        """Return the violations that must be logged for this frame."""
        events: List[ViolationEvent] = []
        worker_by_id = {w.track_id: w for w in workers}

        for worker in workers:
            wid = worker.track_id
            result = compliance.get(wid)
            in_zone = wid in zone_hits

            candidates: List[str] = []

            # --- restricted zone ------------------------------------
            if self.enabled_types[RESTRICTED_ZONE_ENTRY] and in_zone:
                candidates.append(RESTRICTED_ZONE_ENTRY)
            else:
                # keep the state machine fed so leaving resets it
                self._should_fire(wid, RESTRICTED_ZONE_ENTRY, False, now)

            # --- PPE (only meaningful with a real PPE model) ---------
            if result is not None and ppe_model_available:
                for item, worn in result.items.items():
                    vtype = MISSING_TYPE_BY_ITEM.get(item)
                    if vtype is None or not self.enabled_types.get(vtype, False):
                        continue
                    if not worn:
                        candidates.append(vtype)
                    else:
                        self._should_fire(wid, vtype, False, now)

                if result.status == PARTIAL and self.enabled_types[PPE_PARTIAL]:
                    candidates.append(PPE_PARTIAL)
                else:
                    self._should_fire(wid, PPE_PARTIAL, False, now)

                if result.status == UNSAFE and self.enabled_types[PPE_UNSAFE]:
                    candidates.append(PPE_UNSAFE)
                else:
                    self._should_fire(wid, PPE_UNSAFE, False, now)

            for vtype in candidates:
                if not self._should_fire(wid, vtype, True, now):
                    continue
                events.append(self._build_event(worker, result, in_zone, vtype,
                                                video_time_seconds, frame_index))
                if vtype == RESTRICTED_ZONE_ENTRY:
                    self.zone_entry_count += 1

        # forget state for workers that vanished
        if len(self._states) > 4000:
            self._states = {k: v for k, v in self._states.items()
                            if k[0] in worker_by_id}
        return events

    # ------------------------------------------------------------------
    def _build_event(self, worker: TrackedWorker,
                     result: Optional[ComplianceResult],
                     in_zone: bool, vtype: str,
                     video_time_seconds: float,
                     frame_index: int) -> ViolationEvent:
        items = result.items if result else {}
        status = result.status if result else UNSAFE
        conf = result.confidence if result else 0.0
        if vtype == RESTRICTED_ZONE_ENTRY:
            conf = max(conf, round(float(worker.confidence), 3))
        return ViolationEvent(
            event_id=self.logger.next_event_id(),
            timestamp=timestamp_now(),
            video_time=seconds_to_hms(video_time_seconds),
            worker_id=worker.track_id,
            violation_type=vtype,
            helmet="YES" if items.get("helmet") else "NO",
            vest="YES" if items.get("vest") else "NO",
            gloves="YES" if items.get("gloves") else "NO",
            ppe_status=status,
            restricted_zone="YES" if in_zone else "NO",
            confidence=round(float(conf), 3),
            snapshot_path="",
            source=self.source,
            bbox=tuple(int(v) for v in worker.bbox),
            frame_index=frame_index,
        )

    # ------------------------------------------------------------------
    def commit(self, events: Sequence[ViolationEvent], annotated_frame) -> None:
        """Save evidence snapshots (annotated frame) and write CSV rows."""
        for event in events:
            if self.save_snapshot and annotated_frame is not None:
                event.snapshot_path = self.logger.save_snapshot(
                    annotated_frame, event.worker_id, event.violation_type)
            self.logger.log(event)
            print(f"[VIOLATION] #{event.event_id} Worker #{event.worker_id} "
                  f"{event.violation_type} @ {event.video_time} "
                  f"({event.ppe_status})"
                  + (f" -> {event.snapshot_path}" if event.snapshot_path else ""))
