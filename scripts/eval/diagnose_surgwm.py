#!/usr/bin/env python
"""Does the SurgWM re-implementation use its latent actions? Diagnostics on the held-out SurgVU windows of
eval_surgwm.py, on the first 16 frames of each window (the clip length its three models were trained on).

* actions: how often each of the 12 actions is inferred over all transitions, their perplexity, and the number of
  distinct actions per clip; how far apart the 12 codes lie (largest distance between two codes over their mean
  norm, against the median for the tokenizer's 1,024 codes). Perplexity over training batches cannot show a
  collapse: codes that fall out of use restart from the batch's encoder outputs, near copies of the used one.
* action model: its error in rebuilding each next frame (40x60, in [0, 1]) with the inferred actions, with another
  window's, and of copying the previous frame. An action that carries the frame change lowers the first below the
  second. On the windows of the dynamics test also with each of the 12 codes as every action (the largest change
  of a window's error over the codes) and with zero vectors, which no code is near (does the decoder read its
  action input at all?).
* dynamics: the token cross-entropy of the MaskGIT dynamics for each frame after the 5 context frames when all of
  its tokens are masked and every earlier frame is real (the first decoding step of generation, teacher-forced),
  with the inferred actions and with another window's; and the share of a frame's tokens that equal the previous
  frame's. A dynamics model that uses the actions predicts worse with another window's.

    python scripts/eval/diagnose_surgwm.py --device cpu       # -> outputs/eval/surgwm/diagnose.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_surgwm import inputs  # noqa: E402
from eval_wm import EvalWindows, spread  # noqa: E402

from swm.baselines.surgwm import (  # noqa: E402
    LatentActionModel,
    MaskGITDynamics,
    SurgWMConfig,
    VideoVQVAE,
    patchify,
    unpatchify,
)
from swm.data.windows import CACHE  # noqa: E402
from swm.paths import OUTPUTS  # noqa: E402


def load(cls, c: SurgWMConfig, name: str, stage: str, device: str):
    st = torch.load(OUTPUTS / "runs" / f"{name}_{stage}" / "ckpt_last.pt", map_location=device, weights_only=False)
    model = cls(c).to(device).eval()
    model.load_state_dict(st["model"])
    return model, st["step"]


def rebuild_error(lam: LatentActionModel, small: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The action model's decoder (forward()) with the given actions -> squared error per transition, (B, T-1)."""
    c = lam.c
    tok = lam.inp(patchify(small[:, :-1], c.patch))
    y = lam.dec(torch.cat([lam.act_in(q)[:, :, None].to(tok.dtype), tok], 2))[:, :, 1:]
    pred = torch.sigmoid(unpatchify(lam.out(y), c.patch, c.lam_height, c.lam_width).float())
    return (pred - small[:, 1:].float().add(1).div(2)).pow(2).mean((2, 3, 4))


def masked_frame_ce(dyn: MaskGITDynamics, codes: torch.Tensor, acts: torch.Tensor, n_ctx: int) -> tuple:
    """Cross-entropy and accuracy per window and frame n_ctx.. of a fully masked frame given the real frames before
    it, (B, T - n_ctx) each: frame_logits on the masked frame, then the real frame's keys and values over it."""
    b, t, s = codes.shape
    cache, ce, acc = dyn.new_cache(t), [], []
    masked = torch.full_like(codes[:, 0], dyn.mask_id)
    for f in range(t):
        if f >= n_ctx:
            lg = dyn.frame_logits(masked, acts, f, cache).float()                      # (B, S, n_codes)
            ce.append(F.cross_entropy(lg.transpose(1, 2), codes[:, f], reduction="none").mean(1))
            acc.append((lg.argmax(-1) == codes[:, f]).float().mean(1))
        dyn.frame_logits(codes[:, f], acts, f, cache)
    return torch.stack(ce, 1), torch.stack(acc, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="surgwm")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=1000, help="windows for the action statistics and the action model")
    ap.add_argument("--dyn", type=int, default=96, help="windows (spread over the test cases) for the dynamics")
    ap.add_argument("--n-ctx", type=int, default=5)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    c = SurgWMConfig()
    tok, s_tok = load(VideoVQVAE, c, args.name, "tokenizer", args.device)
    lam, s_lam = load(LatentActionModel, c, args.name, "lam", args.device)
    dyn, s_dyn = load(MaskGITDynamics, c, args.name, "dynamics", args.device)
    ds = EvalWindows(args.split)
    real_all = np.load(CACHE / f"eval_frames_{args.split}.npy", mmap_mode="r")
    idx = spread(len(ds), args.limit)
    dyn_idx = set(spread(len(idx), args.dyn).tolist())                    # positions in idx
    case = np.array([ds.parts[ds.index[i, 0]]["stem"].rsplit("_p", 1)[0] for i in idx])
    per = {k: [] for k in ("own", "shuffled", "copy")}
    dyn_per = {k: [] for k in ("ce_own", "ce_shuffled", "acc_own", "acc_shuffled", "same", "code_change",
                               "zero_change")}
    counts, distinct, dyn_pos = torch.zeros(c.n_actions), [], []
    t0 = time.time()
    for s in range(0, len(idx), args.bs):
        pos = np.arange(s, min(s + args.bs, len(idx)))
        real = torch.from_numpy(np.ascontiguousarray(real_all[idx[pos], :c.clip])).to(args.device).float() / 255
        x, small = inputs(real, c)
        with torch.no_grad():
            a, q, _ = lam.actions(small)                                              # (B, 15), (B, 15, d)
            counts += torch.bincount(a.flatten().cpu(), minlength=c.n_actions).float()
            distinct += [len(set(r.tolist())) for r in a.cpu()]
            per["own"].append(rebuild_error(lam, small, q).mean(1).cpu())
            per["shuffled"].append(rebuild_error(lam, small, q.roll(1, 0)).mean(1).cpu())
            per["copy"].append((small[:, :-1] - small[:, 1:]).float().div(2).pow(2).mean((1, 2, 3, 4)).cpu())
            sel = [j for j, p in enumerate(pos) if p in dyn_idx]
            if sel and len(pos) > 1:
                own = rebuild_error(lam, small[sel], q[sel]).mean(1)
                by_code = torch.stack([rebuild_error(lam, small[sel], lam.vq.embed[k].expand_as(q[sel])).mean(1)
                                       for k in range(c.n_actions)], 1)                # (n, 12)
                dyn_per["code_change"].append(((by_code.amax(1) - by_code.amin(1)) / own).cpu())
                zero = rebuild_error(lam, small[sel], torch.zeros_like(q[sel])).mean(1)
                dyn_per["zero_change"].append(((zero - own).abs() / own).cpu())
                codes, _, _ = tok.encode(x[sel])
                acts = torch.cat([torch.full_like(a[sel, :1], c.n_actions), a[sel]], 1)
                other = a.roll(1, 0)[sel]                                             # another window's, as above
                shuf = torch.cat([acts[:, :1], other], 1)
                for k, ak in (("own", acts), ("shuffled", shuf)):
                    ce, acc = masked_frame_ce(dyn, codes, ak, args.n_ctx)
                    dyn_per[f"ce_{k}"].append(ce.mean(1).cpu())
                    dyn_per[f"acc_{k}"].append(acc.mean(1).cpu())
                dyn_per["same"].append((codes[:, args.n_ctx:] == codes[:, args.n_ctx - 1:-1]).float()
                                       .mean((1, 2)).cpu())
                dyn_pos += pos[sel].tolist()
        print(f"{pos[-1] + 1}/{len(idx)} windows ({len(dyn_pos)} for the dynamics), {time.time() - t0:.0f}s",
              flush=True)
    per = {k: torch.cat(v).numpy() for k, v in per.items()}
    dyn_per = {k: torch.cat(v).numpy() for k, v in dyn_per.items()}
    p = counts / counts.sum()
    cases = sorted(set(case))

    def case_wins(lo: np.ndarray, hi: np.ndarray, where: np.ndarray) -> int:
        """The cases whose mean of `lo` is below their mean of `hi`."""
        return int(sum(lo[where == k].mean() < hi[where == k].mean() for k in cases if (where == k).any()))

    def spread_pct(e: torch.Tensor, q: float) -> float:
        """The q-quantile of the distances between two different codes, in % of the mean code norm."""
        d = torch.cdist(e.float(), e.float())[~torch.eye(len(e), dtype=torch.bool, device=e.device)]
        return float(100 * d.quantile(q) / e.float().norm(dim=1).mean())

    dyn_case = case[dyn_pos]
    summary = dict(
        step={"tokenizer": s_tok, "lam": s_lam, "dynamics": s_dyn}, split=args.split, n_windows=len(idx),
        n_cases=len(cases), clip=c.clip,
        action_counts=counts.int().tolist(), action_perplexity=float(torch.exp(-(p * (p + 1e-10).log()).sum())),
        distinct_actions_per_clip=float(np.mean(distinct)),
        code_spread_max_pct=spread_pct(lam.vq.embed, 1.0),
        tokenizer_code_spread_median_pct=spread_pct(tok.vq.embed, 0.5),
        lam_code_change_max_pct=float(100 * dyn_per["code_change"].max()),
        lam_zero_change_mean_pct=float(100 * dyn_per["zero_change"].mean()),
        lam_error={k: float(v.mean()) for k, v in per.items()},
        lam_swap_rise_pct=float(100 * (per["shuffled"].mean() / per["own"].mean() - 1)),
        lam_own_below_copy_pct=float(100 * (1 - per["own"].mean() / per["copy"].mean())),
        lam_cases_own_better=case_wins(per["own"], per["shuffled"], case),
        dyn_windows=len(dyn_pos), dyn_cases=len(set(dyn_case)), n_ctx=args.n_ctx,
        dyn_ce={k: float(dyn_per[f"ce_{k}"].mean()) for k in ("own", "shuffled")},
        dyn_acc={k: float(dyn_per[f"acc_{k}"].mean()) for k in ("own", "shuffled")},
        dyn_ce_swap_rise_pct=float(100 * (dyn_per["ce_shuffled"].mean() / dyn_per["ce_own"].mean() - 1)),
        dyn_cases_own_better=case_wins(dyn_per["ce_own"], dyn_per["ce_shuffled"], dyn_case),
        tokens_same_as_previous=float(dyn_per["same"].mean()), seconds=round(time.time() - t0))
    out = OUTPUTS / "eval" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnose.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out / "diagnose.npz", window=idx, case=case, dyn_window=idx[dyn_pos],
                        **{f"lam_{k}": v for k, v in per.items()}, **{f"dyn_{k}": v for k, v in dyn_per.items()})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
