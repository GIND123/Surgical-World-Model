#!/usr/bin/env python
"""Publish finished world-model runs to the public Hugging Face model repo (token handling in src/swm/hub.py).

Each run becomes one folder of the repo: model.safetensors (the EMA weights that every evaluation uses, fp32,
without the optimiser state), config.json (model and training config, step, parameter count) and the evaluation
summaries that exist for it (replay.json, control.json, action_use.json). The checkpoint tables of the model card
are then rebuilt from the folders in the repo: SurgVU models with the rows of docs/results/e1_main.json, models of
the kinematic benchmarks (E2, E3) with the evaluation on their own dataset. Runs on the CPU, so it can run next to
the GPU queue.

    python scripts/release/push_weights.py S_arms S_arms_scene S_none [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_model

from swm import hub
from swm.models.world_model import WMConfig, WorldModel
from swm.paths import CHECKPOINTS, OUTPUTS, ROOT

STAGE = CHECKPOINTS / "hf"
GITHUB = "https://github.com/GIND123/Surgical-World-Model"
CARD_START, CARD_END = "## Checkpoints", "## Training data"


def export(run: str) -> tuple[Path, int]:
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


def card(text: str, runs: set[str]) -> str:
    """The model card with its checkpoint table rebuilt for `runs` (in the order of the E1 table)."""
    rows = json.loads((ROOT / "docs" / "results" / "e1_main.json").read_text())
    order = [r["run"] for r in rows]
    rows = {r["run"]: r for r in rows}

    def val(r: dict, k: str, fmt: str) -> str:
        return fmt.format(r[k]) if isinstance(r.get(k), (int, float)) else "–"

    train = {run: json.loads((OUTPUTS / "runs" / run / "config.json").read_text())["train"] for run in runs}
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
        r = rows.get(run, {})
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
              'load_model(model, "S_arms/model.safetensors")', "```", "", ""]
    i, j = text.index(CARD_START), text.index(CARD_END)
    return text[:i] + "\n".join(lines) + text[j:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dry-run", action="store_true", help="export and print the model card; upload nothing")
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    api, rid = HfApi(token=hub.token()), hub.repo_id()
    for run in args.runs:
        out, step = export(run)
        size = (out / "model.safetensors").stat().st_size / 1e6
        print(f"{run}: step {step}, {size:.0f} MB -> {out}", flush=True)
        if not args.dry_run:
            hub.push_folder(out, run, f"{run}: EMA weights at step {step}, config and evaluation")
    have = {f.split("/")[0] for f in api.list_repo_files(rid) if f.endswith("/model.safetensors")}
    text = card(Path(hf_hub_download(rid, "README.md", token=hub.token())).read_text(), have | set(args.runs))
    (STAGE / "README.md").write_text(text)
    if args.dry_run:
        print(text)
        return
    hub.push_file(STAGE / "README.md", "README.md", "Model card: checkpoint table")
    print(f"https://huggingface.co/{rid}")


if __name__ == "__main__":
    main()
