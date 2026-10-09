"""
AI-Based Workplace Safety Monitoring and Violation Management System.

Pipeline:
    Camera / Video -> OpenCV -> YOLOv8 -> Person + PPE detection
    -> PPE-to-person matching -> DeepSORT tracking -> Compliance analysis
    -> Restricted zone monitoring -> Violation detection -> Logs + snapshots
"""

__version__ = "1.0.0"
__all__ = [
    "utils",
    "detector",
    "tracker",
    "ppe_matcher",
    "compliance",
    "restricted_zone",
    "violation_manager",
    "logger",
    "visualizer",
]
