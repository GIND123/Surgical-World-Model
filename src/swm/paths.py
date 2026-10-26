"""Canonical locations of data, checkpoints and outputs (all Git-ignored except the code)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
CHECKPOINTS = ROOT / "checkpoints"

SURGVU24_VIDEOS = RAW / "surgvu24" / "videos"
SURGVU24_LABELS = RAW / "surgvu24" / "labels"
SURGVU25_LABELS = RAW / "surgvu25" / "labels"
DETECTOR = CHECKPOINTS / "detector" / "yolo11l_r2_l832.pt"
