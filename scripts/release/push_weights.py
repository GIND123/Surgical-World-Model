#!/usr/bin/env python
"""Publish finished world-model runs to the public Hugging Face model repo (token handling in src/swm/hub.py).

Each run becomes one folder of the repo: model.safetensors (the EMA weights that every evaluation uses, fp32,
without the optimiser state), config.json (model and training config, step, parameter count) and the evaluation
summaries that exist for it (replay.json, control.json, action_use.json). The checkpoint tables of the model card
are then rebuilt from the folders in the repo: SurgVU models with the rows of docs/results/e1_main.json, models of
the kinematic benchmarks (E2, E3) with the evaluation on their own dataset. Runs that a finished queue job was to
publish but that are not in the repo (an upload during a network outage) are published as well. Runs on the CPU, so
it can run next to the GPU queue.

    python scripts/release/push_weights.py S_arms S_arms_scene S_none [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_model

from swm import hub
from swm.baselines.surgwm import LatentActionModel, MaskGITDynamics, SurgWMConfig, VideoVQVAE
from swm.models.world_model import WMConfig, WorldModel
from swm.paths import CHECKPOINTS, OUTPUTS, ROOT

STAGE = CHECKPOINTS / "hf"
GITHUB = "https://github.com/GIND123/Surgical-World-Model"
CARD_START, CARD_END = "## Checkpoints", "## Training data"


SURGWM_STAGES = (("tokenizer", VideoVQVAE), ("lam", LatentActionModel), ("dynamics", MaskGITDynamics))


def export_surgwm(run: str) -> tuple[Path, int]:
    """The SurgWM re-implementation (outputs/runs/<run>_<stage>): one safetensors file per stage, with the final
    parameters (it keeps no moving average) and the quantisers' codebooks."""
    out = STAGE / run
    out.mkdir(parents=True, exist_ok=True)
    meta = {"run": run, "weights": "final parameters of each stage (no moving average)", "step": {}, "parameters": {},
            "train": {}}
    for stage, cls in SURGWM_STAGES:
        st = torch.load(OUTPUTS / "runs" / f"{run}_{stage}" / "ckpt_last.pt", map_location="cpu", weights_only=False)
        conf = json.loads((OUTPUTS / "runs" / f"{run}_{stage}" / "config.json").read_text())
        model = cls(SurgWMConfig(**conf["model"])).eval()
        model.load_state_dict(st["model"])
        save_model(model, str(out / f"{stage}.safetensors"))
        meta["model"] = conf["model"]
        meta["step"][stage], meta["train"][stage] = st["step"], conf["train"]
        meta["parameters"][stage] = sum(p.numel() for p in model.parameters())
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    src = OUTPUTS / "eval" / run / "replay.json"
    if src.exists():
        (out / "replay.json").write_text(src.read_text())
    return out, meta["step"]["dynamics"]


def export(run: str) -> tuple[Path, int]:
    if run.startswith("surgwm"):
        return export_surgwm(run)
    st = torch.load(OUTPUTS / "runs" / run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    cfg = WMConfig.from_dict(st["config"])
    model = WorldModel(cfg).eval()
    model.load_state_dict(st["model"])
    for p, e in zip(model.parameters(), st["ema"], strict=True):
        p.data.copy_(e)
    out = STAGE / run
    out.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out / "model.safetensors"))
    train = json.loads((OUTPUTS / "runs" / run / "config.json").read_text())["train"]
    meta = {"run": run, "step": st["step"], "weights": f"EMA of the parameters (decay {train['ema']})",
            "parameters": sum(p.numel() for p in model.parameters()), "model": cfg.to_dict(), "train": train}
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    ev = OUTPUTS / "eval"
    own = ev / run if train["data"] == "surgvu" else ev / run / train["data"]   # eval_wm.py --data <external>
    for src in (own / "replay.json", own / "control.json", ev / "action_use" / f"{run}_{st['step']}.json"):
        if src.exists():
            summary = json.loads(src.read_text())
            summary.pop("ckpt", None)                                   # local path
            name = "action_use.json" if src.parent.name == "action_use" else src.name
            (out / name).write_text(json.dumps(summary, indent=1))
    return out, st["step"]


def queued() -> set[str]:
    """Runs published by the queue jobs that have finished."""
    return {run for f in (OUTPUTS / "queue" / "done").glob("*.sh")
            for run in re.findall(r"push_weights\.py ([\w.-]+)", f.read_text())}


def card(text: str, runs: set[str]) -> str:
    """The model card with its checkpoint table rebuilt for `runs` (in the order of the E1 and E6 tables; other
    runs, such as further seeds, after them with their own evaluation summaries)."""
    rows = [r for name in ("e1_main", "e6_ablations") if (ROOT / "docs" / "results" / f"{name}.json").exists()
            for r in json.loads((ROOT / "docs" / "results" / f"{name}.json").read_text())]
    order = [r["run"] for r in rows]
    rows = {r["run"]: r for r in rows}
    for run in set(runs) - set(rows):
        r = {}
        for name in ("replay", "control"):
            p = OUTPUTS / "eval" / run / f"{name}.json"
            if p.exists():
                r.update(json.loads(p.read_text()))
        gains = [v for k, v in r.items() if k.startswith("cam_") and k.endswith("_gain")]
        if gains:
            r["cam_gain_mean"] = sum(gains) / len(gains)
        rows[run] = r

    def val(r: dict, k: str, fmt: str) -> str:
        return fmt.format(r[k]) if isinstance(r.get(k), (int, float)) else "–"

    def run_train(run: str) -> dict:
        if run.startswith("surgwm"):                                  # three stages; see export_surgwm
            return {"data": "surgvu", "size": "SurgWM"}
        return json.loads((OUTPUTS / "runs" / run / "config.json").read_text())["train"]

    train = {run: run_train(run) for run in runs}
    for r in rows.values():
        if isinstance(r.get("step"), dict):                           # SurgWM: steps per stage
            r["step"] = "/".join(str(v) for v in r["step"].values())
    other = sorted(run for run in runs if train[run]["data"] != "surgvu")
    lines = [CARD_START, "",
             "One folder per finished run: `model.safetensors` (EMA weights), `config.json` (model and training "
             "config, step) and the evaluation summaries (`replay.json`: fidelity; `control.json`: endoscope "
             "control, locality, embodiment consistency; `action_use.json`: action-use diagnostics). Metrics on "
             f"the held-out SurgVU test cases, 2.8 s horizon, action replay ([protocol]({GITHUB}"
             "#experiments-and-evaluation-protocol)).", "",
             "| Folder | Model | Size | Steps | PSNR ↑ | LPIPS ↓ | FVD ↓ | Endoscope gain ↑ |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for run in sorted(set(runs) - set(other), key=lambda r: (order.index(r) if r in order else len(order), r)):
        r = rows[run]
        lines.append(f"| `{run}/` | {r.get('label', run)} | {train[run]['size']} | {r.get('step', '–')} | "
                     f"{val(r, 'pred_psnr', '{:.2f}')} | {val(r, 'pred_lpips', '{:.3f}')} | "
                     f"{val(r, 'pred_fvd', '{:.0f}')} | {val(r, 'cam_gain_mean', '{:.2f}')} |")
    if other:
        lines += ["", "Checkpoints for the kinematic benchmarks: SurgVU models adapted to a dataset without its "
                  "kinematics (E2, frozen for the linear probes), and models conditioned on the robot kinematics "
                  "(`kin`, E3). Metrics on the test episodes of their own dataset.", "",
                  "| Folder | Data | Actions | Initialised from | Steps | PSNR ↑ | LPIPS ↓ | FVD ↓ |",
                  "|---|---|---|---|---:|---:|---:|---:|"]
        for run in other:
            t = train[run]
            p = OUTPUTS / "eval" / run / t["data"] / "replay.json"
            r = json.loads(p.read_text()) if p.exists() else {}
            share = f", {t['train_fraction']:.0%} of the training episodes" if t["train_fraction"] < 1 else ""
            lines.append(f"| `{run}/` | {t['data']}{share} | `{t['action']}` | {t['init'] or 'scratch'} | "
                         f"{r.get('step', t['steps'])} | {val(r, 'pred_psnr', '{:.2f}')} | "
                         f"{val(r, 'pred_lpips', '{:.3f}')} | {val(r, 'pred_fvd', '{:.0f}')} |")
    lines += ["", f"Loading (with the `swm` package from the [code repository]({GITHUB})):", "", "```python",
              "import json", "from safetensors.torch import load_model",
              "from swm.models.world_model import WMConfig, WorldModel", "",
              'meta = json.load(open("S_arms/config.json"))',
              'model = WorldModel(WMConfig.from_dict(meta["model"])).eval()',
              'load_model(model, "S_arms/model.safetensors")', "```", "",
              "The SurgWM re-implementation (`surgwm/`) has one file per stage (`tokenizer`, `lam`, `dynamics`), "
              "loaded into the classes of `swm.baselines.surgwm` with `SurgWMConfig(**meta[\"model\"])`.", "", ""]
    i, j = text.index(CARD_START), text.index(CARD_END)
    return text[:i] + "\n".join(lines) + text[j:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dry-run", action="store_true", help="export and print the model card; upload nothing")
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    api, rid = HfApi(token=hub.token()), hub.repo_id()
    have = {f.split("/")[0] for f in api.list_repo_files(rid) if "/" in f and f.endswith(".safetensors")}
    runs = list(args.runs) + sorted(queued() - have - set(args.runs))
    for run in runs:
        out, step = export(run)
        size = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e6
        print(f"{run}: step {step}, {size:.0f} MB -> {out}", flush=True)
        if not args.dry_run:
            hub.push_folder(out, run, f"{run}: weights at step {step}, config and evaluation")
    text = card(Path(hf_hub_download(rid, "README.md", token=hub.token())).read_text(), have | set(runs))
    (STAGE / "README.md").write_text(text)
    if args.dry_run:
        print(text)
        return
    hub.push_file(STAGE / "README.md", "README.md", "Model card: checkpoint table")
    print(f"https://huggingface.co/{rid}")


if __name__ == "__main__":
    main()
