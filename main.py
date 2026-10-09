"""
AI-BASED WORKPLACE SAFETY MONITORING AND VIOLATION MANAGEMENT SYSTEM
====================================================================

Camera / Video -> OpenCV -> YOLOv8 -> Person + PPE detection
-> PPE-to-person matching -> DeepSORT tracking -> PPE compliance analysis
-> Restricted zone monitoring -> Violation detection -> Logs + evidence

Usage (Windows CMD):
    python main.py --source videos/input.mp4
    python main.py --source 0
    python main.py --source videos/site.avi --no-display
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import Optional, Tuple

import cv2
import numpy as np

from src.compliance import ComplianceAnalyzer, summarise
from src.detector import build_person_detector, build_ppe_detector
from src.logger import ViolationLogger
from src.ppe_matcher import PPEMatcher
from src.restricted_zone import RestrictedZoneManager
from src.tracker import WorkerTracker
from src.utils import (
    FPSMeter,
    device_label,
    ensure_output_dirs,
    file_timestamp,
    load_config,
    print_banner,
    resolve_path,
    select_device,
)
from src.violation_manager import ViolationManager
from src.visualizer import Visualizer

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
HELP_DISPLAY_SECONDS = 6.0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Based Workplace Safety Monitoring System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default=None,
                        help="Video file path or webcam index (e.g. 0). "
                             "Defaults to videos/input.mp4")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", default=None,
                        help="Output video filename (inside output/processed)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no OpenCV window)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write the processed output video")
    parser.add_argument("--select-zone", action="store_true",
                        help="Force interactive restricted-zone selection")
    parser.add_argument("--no-zone", action="store_true",
                        help="Disable restricted-zone monitoring for this run")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Video source
# --------------------------------------------------------------------------
def resolve_source(source_arg: Optional[str]) -> Tuple[object, str]:
    """Return (cv2 source, human readable name). Exits with a clear message."""
    if source_arg is None:
        default_video = resolve_path("videos/input.mp4")
        if not default_video.exists():
            print_banner([
                "NO VIDEO SOURCE PROVIDED",
                "",
                f"Expected default video at: {default_video}",
                "",
                "Fix it in one of these ways:",
                "  1) Copy a video to videos/input.mp4",
                "  2) python main.py --source path/to/your_video.mp4",
                "  3) python main.py --source 0     (webcam)",
            ])
            sys.exit(1)
        return str(default_video), str(default_video)

    if str(source_arg).isdigit():
        return int(source_arg), f"webcam:{source_arg}"

    path = resolve_path(source_arg)
    if not path.exists():
        print_banner([
            "VIDEO FILE NOT FOUND",
            f"Looked for: {path}",
            "Check the path, or use --source 0 for the webcam.",
        ])
        sys.exit(1)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[INPUT] Warning: '{path.suffix}' is unusual. "
              f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    return str(path), str(path)


def open_capture(source) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if isinstance(source, int):
            print_banner([
                "CAMERA COULD NOT BE OPENED",
                f"Camera index {source} is unavailable.",
                "Close other apps using the webcam, or try --source 1",
            ])
        else:
            print_banner([
                "VIDEO COULD NOT BE OPENED",
                f"{source}",
                "The file may be corrupted or use an unsupported codec.",
                "Try re-encoding it to H.264 MP4.",
            ])
        sys.exit(1)
    return cap


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
class SafetyMonitorApp:
    """Wires every module together and runs the frame loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = load_config(args.config)
        ensure_output_dirs(self.config)

        if args.no_zone:
            self.config["restricted_zone"]["enabled"] = False
        if args.no_save:
            self.config["output"]["save_video"] = False
        if args.no_display:
            self.config["display"]["show_window"] = False

        use_cuda = self.config["device"]["use_cuda_if_available"] and not args.cpu
        self.device = select_device(use_cuda)

        self.source, self.source_name = resolve_source(args.source)
        self.cap = open_capture(self.source)

        # --- video properties -----------------------------------------
        self.src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.src_fps <= 1.0 or self.src_fps > 240:
            self.src_fps = 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.is_stream = isinstance(self.source, int) or self.total_frames <= 0

        # --- models ----------------------------------------------------
        print(f"[DEVICE] Device: {device_label(self.device)}")
        print("[MODEL] Loading person detector ...")
        self.person_detector = build_person_detector(self.config, self.device)
        print("[MODEL] Loading PPE detector ...")
        self.ppe_detector = build_ppe_detector(self.config, self.device)
        self.ppe_available = bool(getattr(self.ppe_detector, "available", False))

        # --- pipeline components ---------------------------------------
        self.tracker = WorkerTracker(self.config, self.device)
        self.matcher = PPEMatcher(self.config)
        self.analyzer = ComplianceAnalyzer(self.config)
        self.zones = RestrictedZoneManager(self.config)
        self.logger = ViolationLogger(self.config, source_name=self.source_name)
        self.violations = ViolationManager(self.config, self.logger,
                                           source=self.source_name)
        self.visualizer = Visualizer(self.config)

        self.resize_width = self.config["display"].get("resize_width")
        self.window_name = self.config["display"].get("window_name",
                                                      "Workplace Safety Monitor")
        self.show_window = bool(self.config["display"].get("show_window", True))
        self.writer: Optional[cv2.VideoWriter] = None
        self.fps_meter = FPSMeter(30)
        self.frame_index = 0
        self.start_time = time.time()

    # ------------------------------------------------------------------
    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optional downscale so inference + display stay fast."""
        if not self.resize_width:
            return frame
        width = int(self.resize_width)
        if frame.shape[1] <= width:
            return frame
        scale = width / float(frame.shape[1])
        return cv2.resize(frame, (width, int(frame.shape[0] * scale)),
                          interpolation=cv2.INTER_LINEAR)

    def _init_writer(self, frame: np.ndarray) -> None:
        if not self.config["output"].get("save_video", True) or self.writer is not None:
            return
        out_dir = resolve_path(self.config["output"]["processed_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        name = self.args.output or self.config["output"].get("output_video_name",
                                                             "result.mp4")
        out_path = out_dir / name
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self.src_fps, (w, h))
        if not writer.isOpened():
            print(f"[OUTPUT] Could not open video writer for {out_path} - "
                  "processed video will not be saved.")
            self.config["output"]["save_video"] = False
            return
        self.writer = writer
        print(f"[OUTPUT] Writing processed video -> {out_path}")

    def _video_time(self) -> float:
        if self.is_stream:
            return time.time() - self.start_time
        pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
        if pos_ms > 0:
            return pos_ms / 1000.0
        return self.frame_index / self.src_fps

    # ------------------------------------------------------------------
    def setup_zones(self, first_frame: np.ndarray) -> None:
        if not self.config["restricted_zone"].get("enabled", True):
            return
        if self.show_window:
            self.zones.setup(first_frame, force_interactive=self.args.select_zone)
        else:
            # headless: only file/config zones are possible
            if not (self.zones.load_saved and self.zones.load_from_file()):
                self.zones.load_from_config()
            if not self.zones.zones:
                print("[ZONE] Headless run with no saved zones - zone monitoring off.")

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, paused: bool = False) -> np.ndarray:
        """Run the full pipeline on one frame and return the annotated frame."""
        annotated = frame.copy()

        # 1) DETECTION -------------------------------------------------
        persons = self.person_detector.detect(frame)
        ppe_detections = self.ppe_detector.detect(frame)

        # 2) TRACKING --------------------------------------------------
        workers = self.tracker.update(persons, frame)

        # 3) PPE -> PERSON MATCHING -----------------------------------
        matched = self.matcher.match(workers, ppe_detections)

        # 4) COMPLIANCE (temporally smoothed) -------------------------
        compliance = self.analyzer.analyse_all(matched, self.frame_index)

        # 5) RESTRICTED ZONE ------------------------------------------
        zone_hits = self.zones.check_workers(workers)

        # 6) VIOLATIONS ------------------------------------------------
        video_time = self._video_time()
        events = self.violations.evaluate(
            workers=workers,
            compliance=compliance,
            zone_hits=zone_hits,
            now=time.time(),
            video_time_seconds=video_time,
            frame_index=self.frame_index,
            ppe_model_available=self.ppe_available,
        )

        # 7) VISUALISATION --------------------------------------------
        stats = summarise(list(compliance.values()))
        stats["violations"] = self.logger.total + len(events)
        stats["zone_entries"] = self.violations.zone_entry_count
        help_left = HELP_DISPLAY_SECONDS - (time.time() - self.start_time)

        annotated = self.visualizer.render(
            frame=annotated,
            workers=workers,
            compliance=compliance,
            ppe_detections=ppe_detections,
            zones=self.zones.zones,
            zone_hits=zone_hits,
            stats=stats,
            fps=self.fps_meter.fps,
            video_time=video_time,
            device=device_label(self.device),
            ppe_model_available=self.ppe_available,
            help_seconds_left=help_left,
            paused=paused,
        )

        # 8) EVIDENCE + LOGS (annotated frame, never a raw one) --------
        for event in events:
            evidence = self.visualizer.make_evidence_frame(annotated, event)
            self.violations.commit([event], evidence)

        if self.frame_index % 300 == 0:
            self.analyzer.cleanup(self.frame_index)

        return annotated

    # ------------------------------------------------------------------
    def run(self) -> None:
        print_banner([
            "AI-BASED WORKPLACE SAFETY MONITORING SYSTEM",
            f"Source : {self.source_name}",
            f"Device : {device_label(self.device)}",
            f"PPE    : {'custom model loaded' if self.ppe_available else 'MODEL MISSING'}",
            "",
            "Controls:  Q = Quit | P = Pause/Resume | "
            "R = Reset zone | S = Screenshot",
        ])

        ok, first_frame = self.cap.read()
        if not ok or first_frame is None:
            print("[INPUT] Could not read the first frame from the source.")
            self.cleanup()
            return
        first_frame = self._prepare_frame(first_frame)
        self.setup_zones(first_frame)
        self._init_writer(first_frame)

        if self.show_window:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, min(1280, first_frame.shape[1]),
                                 min(720, first_frame.shape[0]))
            except cv2.error as exc:
                print(f"[DISPLAY] No GUI available ({exc}) - running headless.")
                self.show_window = False

        self.start_time = time.time()
        frame = first_frame
        paused = False
        last_annotated = None
        corrupt_streak = 0

        while True:
            if not paused:
                if self.frame_index > 0:
                    ok, raw = self.cap.read()
                    if not ok or raw is None:
                        corrupt_streak += 1
                        if self.is_stream and corrupt_streak < 10:
                            print("[INPUT] Dropped frame from stream - retrying...")
                            time.sleep(0.05)
                            continue
                        print("[INPUT] End of stream.")
                        break
                    corrupt_streak = 0
                    frame = self._prepare_frame(raw)

                try:
                    last_annotated = self.process_frame(frame, paused=False)
                except cv2.error as exc:
                    print(f"[FRAME] Skipping corrupted frame {self.frame_index}: {exc}")
                    self.frame_index += 1
                    continue

                self.fps_meter.tick()
                if self.writer is not None:
                    self.writer.write(last_annotated)
                self.frame_index += 1

                if self.frame_index % 100 == 0:
                    pct = (f" ({100.0 * self.frame_index / self.total_frames:.0f}%)"
                           if self.total_frames > 0 else "")
                    print(f"[RUN] Frame {self.frame_index}{pct} | "
                          f"FPS {self.fps_meter.fps:.1f} | "
                          f"Violations {self.logger.total}")

            if self.show_window and last_annotated is not None:
                display = last_annotated
                if paused:
                    display = last_annotated.copy()
                    self.visualizer.draw_paused(display)
                cv2.imshow(self.window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[RUN] Quit requested.")
                    break
                if key in (ord("p"), ord("P")):
                    paused = not paused
                    print(f"[RUN] {'Paused' if paused else 'Resumed'}.")
                if key in (ord("r"), ord("R")):
                    print("[RUN] Re-selecting restricted zone ...")
                    self.zones.reset()
                    self.zones.select_interactively(frame.copy())
                if key in (ord("s"), ord("S")):
                    shot_dir = resolve_path(self.config["output"]["snapshots_dir"])
                    shot_dir.mkdir(parents=True, exist_ok=True)
                    shot = shot_dir / f"manual_{file_timestamp()}.jpg"
                    cv2.imwrite(str(shot), last_annotated)
                    print(f"[RUN] Screenshot saved -> {shot}")
            elif paused:
                time.sleep(0.05)

        self.cleanup()

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
        if self.writer is not None:
            self.writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        self.logger.export_json()
        elapsed = max(1e-6, time.time() - self.start_time)
        print_banner([
            "RUN COMPLETE",
            f"Frames processed : {self.frame_index}",
            f"Average FPS      : {self.frame_index / elapsed:.1f}",
            f"Violations logged: {self.logger.total}",
            f"Zone entries     : {self.violations.zone_entry_count}",
            f"CSV log          : {self.logger.csv_path}",
            f"Snapshots        : {self.logger.snapshot_dir}",
        ])
        breakdown = self.logger.summary()
        if breakdown:
            print("Violation breakdown:")
            for vtype, count in sorted(breakdown.items()):
                print(f"   {vtype:<24} {count}")


# --------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    try:
        app = SafetyMonitorApp(args)
        app.run()
    except KeyboardInterrupt:
        print("\n[RUN] Interrupted by user (Ctrl+C).")
    except RuntimeError as exc:
        print("\n[ERROR] " + str(exc))
        return 1
    except FileNotFoundError as exc:
        print(f"\n[ERROR] Missing file: {exc}")
        return 1
    except Exception as exc:  # last-resort friendly handler
        print("\n[ERROR] Unexpected failure:", exc)
        print("-" * 70)
        traceback.print_exc()
        print("-" * 70)
        print("See the Troubleshooting section of README.md.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
