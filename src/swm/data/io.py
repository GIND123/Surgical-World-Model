"""Atomic writes and tolerant reads for files that other pipeline stages poll while they are being produced."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

UNREADABLE = (EOFError, ValueError, OSError, KeyError)


def save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def settled(path: Path, seconds: float = 30.0) -> bool:
    """True when the file has not been modified for ``seconds`` (guards readers against non-atomic writers)."""
    try:
        return time.time() - path.stat().st_mtime > seconds
    except FileNotFoundError:
        return False


def try_load(path: Path):
    import zipfile
    try:
        return np.load(path)
    except (*UNREADABLE, zipfile.BadZipFile):
        return None
