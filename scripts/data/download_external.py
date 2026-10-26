#!/usr/bin/env python
"""Download the openly licensed external datasets into data/external/<name>/ (resumable, checksum-verified).

    python scripts/data/download_external.py open_h_suturebot open_h_srth_chole open_h_crcd sar_rarp50
    python scripts/data/download_external.py --list

Form-gated datasets (JIGSAWS, Cholec80, AutoLaparo) are not handled here; see docs/02_external_data.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from swm.paths import EXTERNAL

OPEN_H = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"     # CC-BY-4.0, LeRobot v2.1
REGISTRY = {
    "open_h_suturebot": dict(kind="hf", repo=OPEN_H, pattern="Surgical/jhu/imerse/suturebot/**",
                             note="dVRK-Si suturing with kinematics; SurgCast benchmark"),
    "open_h_srth_chole": dict(kind="hf", repo=OPEN_H, pattern="Surgical/jhu/imerse/srth_porcine_chole/**",
                              note="dVRK ex-vivo porcine cholecystectomy (SRT-H); SurgCast benchmark"),
    "open_h_crcd": dict(kind="hf", repo=OPEN_H, pattern="Surgical/uic/uic_crcd_lerobot/**",
                        note="da Vinci porcine cholecystectomy with kinematics (CRCD)"),
    "sar_rarp50": dict(kind="figshare", articles=[24932529, 24932499],
                       note="50 clinical robot-assisted prostatectomies, gestures + segmentation; CC BY-NC-SA 4.0"),
}


def hf(name: str, spec: dict) -> None:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    snapshot_download(spec["repo"], repo_type="dataset", allow_patterns=[spec["pattern"]],
                      local_dir=EXTERNAL / name, max_workers=16)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def figshare(name: str, spec: dict) -> None:
    for article in spec["articles"]:
        meta = json.load(urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{article}", timeout=60))
        out = EXTERNAL / name / meta["title"].replace(" ", "_")
        out.mkdir(parents=True, exist_ok=True)
        (out / "figshare_meta.json").write_text(json.dumps(meta, indent=1))
        for f in meta["files"]:
            dst = out / f["name"]
            if dst.exists() and dst.stat().st_size == f["size"] and md5(dst) == f["computed_md5"]:
                continue
            subprocess.run(["curl", "-sSL", "--retry", "8", "--retry-delay", "10", "-C", "-", "-o", str(dst),
                            f["download_url"]], check=True)
            if md5(dst) != f["computed_md5"]:
                dst.unlink()
                raise RuntimeError(f"md5 mismatch for {dst}")
            print(f"ok {dst.relative_to(EXTERNAL)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.names:
        for k, v in REGISTRY.items():
            print(f"{k:20s} {v['note']}")
        return
    for name in args.names:
        spec = REGISTRY[name]
        print(f"== {name}: {spec['note']}", flush=True)
        {"hf": hf, "figshare": figshare}[spec["kind"]](name, spec)
        print(f"== done {name}", flush=True)


if __name__ == "__main__":
    main()
