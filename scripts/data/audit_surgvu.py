#!/usr/bin/env python
"""Audit the local SurgVU data: video integrity and metadata, label coverage and label consistency.

Writes outputs/audit/{videos.csv, cases.csv, tools_intervals.csv, tasks_intervals.csv, summary.json}.
Every number quoted in docs/01_data_audit.md comes from this script.

    python scripts/data/audit_surgvu.py --workers 20
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
VIDEOS = RAW / "surgvu24" / "videos"
LABELS24 = RAW / "surgvu24" / "labels"
LABELS25 = RAW / "surgvu25" / "labels"
OUT = ROOT / "outputs" / "audit"

PART_RE = re.compile(r"case_(\d{3})_video_part_(\d{3})\.mp4$")


def probe(path: str) -> dict:
    """Container metadata plus a decode check at 5%, 50% and 95% of the duration."""
    rec = {"path": path, "bytes": Path(path).stat().st_size}
    m = PART_RE.search(path)
    rec["case"], rec["part"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    try:
        with av.open(path) as c:
            s = c.streams.video[0]
            cc = s.codec_context
            rec.update(codec=cc.name, width=cc.width, height=cc.height, pix_fmt=cc.pix_fmt,
                       fps=float(s.average_rate) if s.average_rate else None,
                       n_frames=s.frames, duration_s=c.duration / 1e6 if c.duration else None,
                       bitrate=c.bit_rate, n_audio=len(c.streams.audio))
            ok, means = 0, []
            for frac in (0.05, 0.5, 0.95):
                c.seek(int(frac * c.duration), any_frame=False, backward=True)
                for frame in c.decode(video=0):
                    img = frame.to_ndarray(format="gray")
                    means.append(float(img.mean()))
                    ok += 1
                    break
            rec["decode_ok"] = ok
            rec["luma_samples"] = means
    except Exception as ex:  # noqa: BLE001 -- any decoder failure means a corrupt or truncated file
        rec["error"] = repr(ex)
    return rec


def hms(x) -> float:
    """'01:02:03.500000' or a float string -> seconds."""
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)
    h, m, s = str(x).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_tools(case_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(case_dir / "tools.csv")
    df["case"] = case_dir.name
    df["t0"] = df["install_case_time"].map(hms)
    df["t1"] = df["uninstall_case_time"].map(hms)
    return df


def load_tasks(case_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(case_dir / "tasks.csv")
    df["case"] = case_dir.name
    df["t0"] = df["start_time"].map(hms)
    df["t1"] = df["stop_time"].map(hms)
    return df


def union_length(intervals) -> float:
    total, end = 0.0, -np.inf
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- videos ---------------------------------------------------------------------------------
    files = sorted(str(p) for p in VIDEOS.rglob("*.mp4"))
    with ProcessPoolExecutor(args.workers) as ex:
        vids = pd.DataFrame(list(ex.map(probe, files, chunksize=4)))
    vids.to_csv(OUT / "videos.csv", index=False)
    stray = sorted(str(p.relative_to(VIDEOS)) for p in VIDEOS.rglob("*") if p.is_file() and p.suffix != ".mp4")
    misnamed = vids[vids["case"].isna()]["path"].tolist()

    # ---- labels ---------------------------------------------------------------------------------
    case_dirs24 = sorted(p for p in LABELS24.iterdir() if p.is_dir())
    tools = pd.concat([load_tools(d) for d in case_dirs24], ignore_index=True)
    tasks = pd.concat([load_tasks(d) for d in case_dirs24], ignore_index=True)
    tools.to_csv(OUT / "tools_intervals.csv", index=False)
    tasks.to_csv(OUT / "tasks_intervals.csv", index=False)
    case_dirs25 = sorted(p for p in LABELS25.iterdir() if p.is_dir())
    tasks25 = pd.concat([load_tasks(d) for d in case_dirs25], ignore_index=True)
    tools25 = pd.concat([load_tools(d) for d in case_dirs25], ignore_index=True)

    # ---- per-case table -------------------------------------------------------------------------
    ok = vids[vids["error"].isna()] if "error" in vids else vids
    rows = []
    video_cases = {f"case_{int(c):03d}" for c in ok["case"].dropna()}
    label_cases = {d.name for d in case_dirs24}
    for case in sorted(video_cases | label_cases):
        cnum = int(case.split("_")[1])
        v = ok[ok["case"] == cnum].sort_values("part")
        tc = tools[tools["case"] == case]
        kc = tasks[tasks["case"] == case]
        part_dur = dict(zip(v["part"].astype(int), v["duration_s"], strict=True))
        # labels give (part, time-within-part); flag any label that runs past its part's end
        over_tools = sum(1 for _, r in tc.iterrows()
                         if part_dur.get(int(r["uninstall_case_part"]), np.inf) < r["t1"] - 1.0)
        over_tasks = sum(1 for _, r in kc.iterrows()
                         if part_dur.get(int(r["stop_part"]), np.inf) < r["t1"] - 1.0)
        cross_part = int((tc["install_case_part"] != tc["uninstall_case_part"]).sum()) + \
            int((kc["start_part"] != kc["stop_part"]).sum())
        tool_cover = sum(union_length([(a, b) for a, b in zip(g["t0"], g["t1"], strict=True)])
                         for _, g in tc.groupby("install_case_part"))
        task_cover = sum(union_length([(a, b) for a, b in zip(g["t0"], g["t1"], strict=True)])
                         for _, g in kc.groupby("start_part"))
        rows.append(dict(
            case=case, n_parts=len(v), video_s=float(v["duration_s"].sum()) if len(v) else 0.0,
            has_video=case in video_cases, has_labels=case in label_cases,
            n_tool_rows=len(tc), n_tool_rows_unique=len(tc.drop_duplicates(subset=[
                "install_case_part", "install_case_time", "uninstall_case_part", "uninstall_case_time", "arm",
                "groundtruth_toolname"])),
            arms=",".join(sorted(tc["arm"].dropna().unique())), n_tools=tc["groundtruth_toolname"].nunique(),
            n_task_rows=len(kc), tasks=";".join(sorted(kc["groundtruth_taskname"].dropna().unique())),
            tool_cover_s=tool_cover, task_cover_s=task_cover, labels_past_part_end=over_tools + over_tasks,
            cross_part_intervals=cross_part))
    cases = pd.DataFrame(rows)
    cases.to_csv(OUT / "cases.csv", index=False)

    # ---- SurgVU24 vs SurgVU25 label agreement ----------------------------------------------------
    key = ["case", "t0", "t1"]
    t24 = tasks[key + ["groundtruth_taskname"]].drop_duplicates()
    t25 = tasks25[key + ["groundtruth_taskname"]].drop_duplicates()
    merged = t24.merge(t25, on=key, how="outer", suffixes=("_24", "_25"), indicator=True)

    tools_u = tools.drop_duplicates(subset=["case", "install_case_part", "t0", "uninstall_case_part", "t1", "arm",
                                            "groundtruth_toolname"])
    tools_u = tools_u.assign(dur=tools_u["t1"] - tools_u["t0"])
    tasks_u = tasks.drop_duplicates(subset=["case", "start_part", "t0", "stop_part", "t1", "groundtruth_taskname"])
    tasks_u = tasks_u.assign(dur=tasks_u["t1"] - tasks_u["t0"])
    desc = tasks25.drop_duplicates(subset=["groundtruth_taskname", "matched_description"])

    summary = dict(
        n_video_files=len(vids), n_video_errors=int(vids["error"].notna().sum()) if "error" in vids else 0,
        errors=vids[vids["error"].notna()][["path", "error"]].to_dict("records") if "error" in vids else [],
        decode_failures=ok[ok["decode_ok"] < 3]["path"].tolist(),
        stray_files=stray, misnamed_videos=misnamed,
        total_hours=float(ok["duration_s"].sum() / 3600), total_tb=float(vids["bytes"].sum() / 1e12),
        resolutions=ok.groupby(["width", "height"]).size().rename("n").reset_index().to_dict("records"),
        fps=ok["fps"].round(3).value_counts().to_dict(), codecs=ok["codec"].value_counts().to_dict(),
        part_hours=ok["duration_s"].describe().div(3600).round(3).to_dict(),
        n_cases_video=len(video_cases), n_cases_labels=len(label_cases),
        cases_video_only=sorted(video_cases - label_cases), cases_labels_only=sorted(label_cases - video_cases),
        parts_per_case=cases[cases["has_video"]]["n_parts"].value_counts().sort_index().to_dict(),
        case_hours=cases[cases["has_video"]]["video_s"].describe().div(3600).round(3).to_dict(),
        tool_rows=len(tools), tool_rows_unique=len(tools_u), tool_duplicate_rows=len(tools) - len(tools_u),
        arms=tools_u["arm"].value_counts().to_dict(),
        tools_by_rows=tools_u["groundtruth_toolname"].str.strip().value_counts().to_dict(),
        tool_hours_by_name=tools_u.assign(n=tools_u["groundtruth_toolname"].str.strip()).groupby("n")["dur"].sum()
        .div(3600).round(2).sort_values(ascending=False).to_dict(),
        tool_hours_by_arm=tools_u.groupby("arm")["dur"].sum().div(3600).round(2).to_dict(),
        tool_arm_table=pd.crosstab(tools_u["groundtruth_toolname"].str.strip(), tools_u["arm"]).to_dict(),
        commercial_names=tools_u.groupby(tools_u["groundtruth_toolname"].str.strip())["commercial_toolname"]
        .unique().map(sorted).to_dict(),
        negative_tool_intervals=int((tools_u["dur"] < 0).sum()),
        task_rows=len(tasks), task_rows_unique=len(tasks_u),
        tasks_by_rows=tasks_u["groundtruth_taskname"].value_counts().to_dict(),
        task_hours_by_name=tasks_u.groupby("groundtruth_taskname")["dur"].sum().div(3600).round(2)
        .sort_values(ascending=False).to_dict(),
        task_minutes=tasks_u["dur"].describe().div(60).round(2).to_dict(),
        negative_task_intervals=int((tasks_u["dur"] < 0).sum()),
        tool_cover_hours=float(cases["tool_cover_s"].sum() / 3600),
        task_cover_hours=float(cases["task_cover_s"].sum() / 3600),
        cases_with_labels_past_part_end=int((cases["labels_past_part_end"] > 0).sum()),
        cross_part_intervals=int(cases["cross_part_intervals"].sum()),
        surgvu25_cases=len(case_dirs25), surgvu25_task_rows=len(tasks25), surgvu25_tool_rows=len(tools25),
        task_label_agreement_24_vs_25=merged["_merge"].value_counts().to_dict(),
        task_name_mismatch_24_vs_25=int((merged["_merge"].eq("both") &
                                         merged["groundtruth_taskname_24"].ne(merged["groundtruth_taskname_25"])).sum()),
        surgvu25_descriptions_per_task=desc.groupby("groundtruth_taskname").size().to_dict(),
        surgvu25_taskname_variants=tasks25.groupby("groundtruth_taskname")["taskname"].unique().map(sorted).to_dict(),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in [
        "n_video_files", "n_video_errors", "decode_failures", "total_hours", "total_tb", "resolutions", "fps",
        "n_cases_video", "n_cases_labels", "cases_video_only", "cases_labels_only", "parts_per_case", "case_hours",
        "tool_rows", "tool_duplicate_rows", "arms", "tool_cover_hours", "task_cover_hours",
        "cases_with_labels_past_part_end", "cross_part_intervals", "task_label_agreement_24_vs_25",
        "task_name_mismatch_24_vs_25", "stray_files", "misnamed_videos"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
