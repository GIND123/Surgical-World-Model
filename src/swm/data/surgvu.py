"""SurgVU robot-log labels on a single per-case timeline.

The organisers' CSVs give every event as (video part, seconds within that part). Parts are consecutive files
of one recording, so a case's global time is ``offset[part] + t`` with ``offset`` the summed durations of
the earlier parts. ``tools.csv`` logs every install/uninstall on the four universal surgical manipulators
(USM1-4), *including the endoscope* (logged as ``nan(camera in)``), so the full robot configuration -- which
arm carries the camera and which instrument sits on each other arm -- is known at every instant.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from swm.paths import PROCESSED, SURGVU24_LABELS, SURGVU24_VIDEOS, SURGVU25_LABELS

ARMS = ("USM1", "USM2", "USM3", "USM4")
EMPTY, CAMERA = "empty", "camera"
# The 14 SurgVU detection classes first (same order as the legacy detector), then rare extras.
DETECTOR_CLASSES = (
    "grasping retractor", "cadiere forceps", "bipolar forceps", "force bipolar", "clip applier", "stapler",
    "permanent cautery hook/spatula", "monopolar curved scissors", "vessel sealer", "tip-up fenestrated grasper",
    "bipolar dissector", "needle driver", "prograsp forceps", "suction irrigator",
)
EXTRA_INSTRUMENTS = ("synchroseal", "curved scissors", "crocodile grasper", "potts scissors", "tenaculum forceps")
UNKNOWN = "unknown"   # logged as "Unknown Instrument" with no ground-truth name
OCCUPANTS = (EMPTY, CAMERA, *DETECTOR_CLASSES, *EXTRA_INSTRUMENTS, UNKNOWN)
OCC_INDEX = {name: i for i, name in enumerate(OCCUPANTS)}
TASKS = ("Range of motion", "Rectal artery/vein", "Retraction and collision avoidance", "Skills application",
         "Suspensory ligaments", "Suturing", "Uterine horn", "Other")
_TASK_CANON = {t.lower(): t for t in TASKS}

PART_RE = re.compile(r"case_(\d{3})_video_part_(\d{3})\.mp4$")
PARTS_CACHE = PROCESSED / "surgvu24_parts.json"


def hms(x) -> float:
    """'01:02:03.5' or a number -> seconds."""
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if ":" not in s:
        return float(s)
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def canonical_instrument(name) -> str:
    if pd.isna(name):
        return UNKNOWN
    n = str(name).strip().lower()
    if n == "nan(camera in)":
        return CAMERA
    if n not in OCC_INDEX:
        raise KeyError(f"unknown instrument name {name!r}")
    return n


def canonical_task(name: str) -> str:
    return _TASK_CANON[str(name).strip().lower()]


def probe_parts(videos: Path = SURGVU24_VIDEOS) -> dict[str, list[dict]]:
    """{case: [{part, path, duration, n_frames, fps}, ...]} from the video containers."""
    import av

    out: dict[str, list[dict]] = {}
    for p in sorted(videos.rglob("*.mp4")):
        m = PART_RE.search(p.name)
        if not m:
            continue
        with av.open(str(p)) as c:
            s = c.streams.video[0]
            out.setdefault(f"case_{m.group(1)}", []).append(dict(
                part=int(m.group(2)), path=str(p.relative_to(videos)), duration=c.duration / 1e6,
                n_frames=s.frames, fps=float(s.average_rate)))
    for parts in out.values():
        parts.sort(key=lambda d: d["part"])
    return out


@lru_cache(maxsize=1)
def parts_table() -> dict[str, list[dict]]:
    """Cached :func:`probe_parts` (data/processed/surgvu24_parts.json)."""
    if not PARTS_CACHE.exists():
        PARTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PARTS_CACHE.write_text(json.dumps(probe_parts(), indent=1))
    return json.loads(PARTS_CACHE.read_text())


def cases() -> list[str]:
    return sorted(parts_table())


@dataclass(frozen=True)
class Timeline:
    case: str
    offsets: dict[int, float]   # part -> global start (s)
    duration: float             # whole case (s)

    def to_global(self, part, t) -> float:
        """NaN for a part with no video (case_071 part 1 was never delivered)."""
        off = self.offsets.get(int(float(part)))
        return np.nan if off is None else off + hms(t)

    def part_start(self, part) -> float:
        return self.offsets.get(int(float(part)), np.nan)

    def to_local(self, t: float) -> tuple[int, float]:
        """Global seconds -> (part, seconds within part)."""
        part = max(p for p, o in self.offsets.items() if o <= t)
        return part, t - self.offsets[part]


def timeline(case: str) -> Timeline:
    parts = parts_table()[case]
    offsets, acc = {}, 0.0
    for d in parts:
        offsets[d["part"]] = acc
        acc += d["duration"]
    return Timeline(case, offsets, acc)


def _clip_missing_parts(df: pd.DataFrame, end_parts: pd.Series, tl: Timeline) -> tuple[pd.DataFrame, dict]:
    """Intervals that end in a delivered part but start in a missing one begin at that part's start;
    intervals that end in a missing part are dropped."""
    qc = dict(rows=len(df), missing_part=int(df["t1"].isna().sum()))
    starts = np.array([tl.part_start(p) for p in end_parts])
    clip = df["t0"].isna() & df["t1"].notna()
    df.loc[clip, "t0"] = starts[clip.to_numpy()]
    return df[df["t1"].notna()].copy(), qc


def load_tools(case: str, labels: Path = SURGVU24_LABELS) -> tuple[pd.DataFrame, dict]:
    """Deduplicated arm-occupancy intervals [arm, occupant, t0, t1] in global seconds, plus QC counts."""
    tl = timeline(case)
    raw = pd.read_csv(labels / case / "tools.csv")
    df = pd.DataFrame(dict(
        arm=raw["arm"].str.strip(),
        occupant=raw["groundtruth_toolname"].map(canonical_instrument),
        commercial=raw["commercial_toolname"].str.strip(),
        t0=[tl.to_global(p, t) for p, t in zip(raw["install_case_part"], raw["install_case_time"], strict=True)],
        t1=[tl.to_global(p, t) for p, t in zip(raw["uninstall_case_part"], raw["uninstall_case_time"], strict=True)],
    ))
    df, qc = _clip_missing_parts(df, raw["uninstall_case_part"], tl)
    n_before = len(df)
    df = df.drop_duplicates(subset=["arm", "occupant", "t0", "t1"]).reset_index(drop=True)
    qc["duplicates"] = n_before - len(df)
    qc["inverted"] = int((df["t1"] <= df["t0"]).sum())
    qc["past_end"] = int((df["t1"] > tl.duration + 1.0).sum())
    df = df[df["t1"] > df["t0"]].sort_values(["arm", "t0"]).reset_index(drop=True)
    # two different occupants on one arm at the same time cannot happen physically
    conflicts = 0
    for _, g in df.groupby("arm"):
        g = g.sort_values("t0")
        prev_end, prev_occ = -np.inf, None
        for t0, t1, occ in zip(g["t0"], g["t1"], g["occupant"], strict=True):
            if t0 < prev_end - 1.0 and occ != prev_occ:
                conflicts += 1
            if t1 > prev_end:
                prev_end, prev_occ = t1, occ
    qc["conflicts"] = conflicts
    return df, qc


def load_tasks(case: str, source: str = "surgvu25") -> pd.DataFrame:
    """Task segments [task, t0, t1, raw_name, description] in global seconds (deduplicated)."""
    tl = timeline(case)
    root = SURGVU25_LABELS if source == "surgvu25" else SURGVU24_LABELS
    raw = pd.read_csv(root / case / "tasks.csv")
    df = pd.DataFrame(dict(
        task=raw["groundtruth_taskname"].map(canonical_task),
        t0=[tl.to_global(p, t) for p, t in zip(raw["start_part"], raw["start_time"], strict=True)],
        t1=[tl.to_global(p, t) for p, t in zip(raw["stop_part"], raw["stop_time"], strict=True)],
        raw_name=raw["taskname" if "taskname" in raw else "groundtruth_taskname"].astype("string").str.strip(),
        description=raw["matched_description"].astype("string").str.strip() if "matched_description" in raw else "",
    ))
    df, _ = _clip_missing_parts(df, raw["stop_part"], tl)
    df = df.drop_duplicates(subset=["task", "t0", "t1"])
    return df[df["t1"] > df["t0"]].sort_values("t0").reset_index(drop=True)


def arm_occupancy(tools: pd.DataFrame, duration: float, hz: float = 1.0) -> np.ndarray:
    """[n_steps, 4] occupant indices (into OCCUPANTS) sampled at ``hz``; later installs win on overlap."""
    n = int(duration * hz)
    occ = np.zeros((n, len(ARMS)), dtype=np.int16)
    for a, arm in enumerate(ARMS):
        for t0, t1, name in tools[tools["arm"] == arm][["t0", "t1", "occupant"]].itertuples(index=False):
            occ[max(0, int(np.ceil(t0 * hz))):min(n, int(t1 * hz) + 1), a] = OCC_INDEX[name]
    return occ


def task_track(tasks: pd.DataFrame, duration: float, hz: float = 1.0) -> np.ndarray:
    """[n_steps] task index (into TASKS) or -1 where no task is annotated."""
    n = int(duration * hz)
    trk = np.full(n, -1, dtype=np.int16)
    for t0, t1, name in tasks[["t0", "t1", "task"]].itertuples(index=False):
        trk[max(0, int(np.ceil(t0 * hz))):min(n, int(t1 * hz) + 1)] = TASKS.index(name)
    return trk
