#!/usr/bin/env python
"""Does a trained world model use its learned actions? Diagnostics on validation windows.

* posterior: active arms, region size, |mu|, posterior variance, KL per arm, active units (Var_x E[u|x] > 0.01)
* use: context-mode flow-matching error (2 clean context latents) per noise level tau with the inferred learned
  actions, with another window's (shuffled), with zero actions and with the camera action shuffled; inside and
  outside the instrument regions. A model that uses the arm actions loses accuracy inside the regions when they
  are shuffled.
* sensitivity: relative change of the predicted velocity when the learned actions are scaled by 3.
* replay: latent error of generated futures (2 context latents, 10 Euler steps, same noise) with the inferred
  actions, with the learned actions of another window and with its camera action.

    python scripts/eval/diagnose_actions.py --run pilot_partial_S_arms --device cpu
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from swm.data.windows import LatentWindows
from swm.models.actions import region_from_boxes
from swm.models.world_model import WMConfig, WorldModel
from swm.paths import OUTPUTS

TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)


def load(run: str, ckpt: str | None, device: str) -> tuple[WorldModel, int]:
    st = torch.load(OUTPUTS / "runs" / run / (ckpt or "ckpt_last.pt"), map_location=device, weights_only=False)
    cfg = WMConfig.from_dict(st["config"])
    model = WorldModel(cfg).to(device).eval()
    for p, e in zip(model.parameters(), st["ema"], strict=True):
        p.data.copy_(e)
    return model, st["step"]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--gen", type=int, default=128, help="windows used for the replay-generation error")
    ap.add_argument("--stems", help="JSON with a 'val' list of parts, so that runs are compared on the same windows")
    args = ap.parse_args()
    torch.manual_seed(0)
    model, step = load(args.run, args.ckpt, args.device)
    stems = json.loads(open(args.stems).read())["val"] if args.stems else None
    loader = DataLoader(LatentWindows("val", max_windows=args.windows, seed=0, stems=stems), batch_size=args.bs,
                        num_workers=2)
    keys = ("real", "shuffled", "zero", "camera_shuffled")
    err = {k: np.zeros((len(TAUS), 2)) for k in keys}          # (tau, inside / outside region)
    cnt = np.zeros((len(TAUS), 2))
    mus, logvars, kls, n_arms, n_active, area, sens = [], [], [], 0, 0, [], []
    gen_keys = ("real", "shuffled", "camera_shuffled")
    gen_err, gen_cnt, n_gen = {k: np.zeros(2) for k in gen_keys}, np.zeros(2), 0
    for bi, batch in enumerate(loader):
        b = {k: v.to(args.device) for k, v in batch.items()}
        h, w = model.cfg.latent_shape[1:]
        b["region"] = region_from_boxes(b["boxes"], h, w, dilate=1)
        z = b["z"]
        bsz, t = z.shape[:2]
        acts = model.infer_actions(z, b, stochastic=False)
        if "u_arms" in acts and "active" in acts:
            act = acts["active"]
            n_arms, n_active = n_arms + act.numel(), n_active + int(act.sum())
            region = b["region"][:, 1:]
            area.append(region[act].float().mean(-1).cpu())
            occ = b["occupant"][:, 1:].flatten(0, 1)
            feats = model.idm.trunk(z[:, :-1].flatten(0, 1), z[:, 1:].flatten(0, 1))
            q = model.idm.arm(torch.arange(4, device=z.device))[None] + model.idm.occ(occ)
            safe = region.flatten(0, 1) | ~act.flatten(0, 1)[..., None]
            mu, lv = model.idm.head(model.idm.pool(q, feats, safe))
            a = act.flatten(0, 1)
            mus.append(mu[a].cpu())
            logvars.append(lv[a].cpu())
            kls.append((0.5 * (mu.pow(2) + lv.exp() - 1 - lv)).sum(-1)[a].cpu())
        inside = b["region"].any(2)[:, 2:]                                  # (B, T-2, H*W) any arm
        shuf = {k: (v.roll(1, 0) if k.startswith("u_") else v) for k, v in acts.items()}
        zero = {k: (torch.zeros_like(v) if k.startswith("u_") else v) for k, v in acts.items()}
        b_cam = dict(b, camera=b["camera"].roll(1, 0))
        ones = torch.ones(bsz, device=z.device)
        for i, tv in enumerate(TAUS):
            tau = torch.full((bsz, t), tv, device=z.device)
            tau[:, :2] = 0
            eps = torch.randn_like(z)
            x = (1 - tv) * z + tv * eps
            x[:, :2] = z[:, :2]
            for key, a, bb in (("real", acts, b), ("shuffled", shuf, b), ("zero", zero, b),
                               ("camera_shuffled", acts, b_cam)):
                tok, mask = model.tokens(bb, a, ones)
                v = model.dit(x, model.cond(tau, bb, ones), tok, mask)
                e = (v - (eps - z)).pow(2).mean(2)[:, 2:].flatten(-2)       # (B, T-2, H*W)
                err[key][i, 0] += float(e[inside].sum())
                err[key][i, 1] += float(e[~inside].sum())
                if key == "real" and "u_arms" in a and i == 2:
                    big = dict(a, u_arms=a["u_arms"] * 3)
                    tok3, _ = model.tokens(bb, big, ones)
                    v3 = model.dit(x, model.cond(tau, bb, ones), tok3, mask)
                    sens.append(float((v3 - v).norm() / v.norm()))
            cnt[i, 0] += int(inside.sum())
            cnt[i, 1] += int((~inside).sum())
        if n_gen < args.gen:
            for key, a, bb in (("real", acts, b), ("shuffled", shuf, b), ("camera_shuffled", acts, b_cam)):
                g = torch.Generator(device=z.device).manual_seed(bi)
                pred = model.generate(bb, n_ctx=2, steps=10, actions=a, generator=g)
                e = (pred[:, 2:] - z[:, 2:]).pow(2).mean(2).flatten(-2)
                gen_err[key] += [float(e[inside].sum()), float(e[~inside].sum())]
            gen_cnt += [int(inside.sum()), int((~inside).sum())]
            n_gen += bsz
    out = {"run": args.run, "step": step, "windows": len(loader.dataset), "val_parts": len(loader.dataset.parts),
           "val_stems": [p["stem"] for p in loader.dataset.parts], "taus": TAUS}
    out["replay"] = {k: {"inside": float(v[0] / gen_cnt[0]), "outside": float(v[1] / gen_cnt[1]),
                         "all": float(v.sum() / gen_cnt.sum())} for k, v in gen_err.items()}
    out["replay_gap_pct"] = {k: round(100 * (out["replay"][k]["all"] / out["replay"]["real"]["all"] - 1), 3)
                             for k in gen_keys[1:]}
    if mus:
        mu, lv, kl = torch.cat(mus), torch.cat(logvars), torch.cat(kls)
        out["posterior"] = dict(active_arm_fraction=n_active / max(n_arms, 1),
                                region_area=float(torch.cat(area).mean()),
                                mean_abs_mu=float(mu.abs().mean()), mean_post_var=float(lv.exp().mean()),
                                kl_per_arm=float(kl.mean()), var_mu_per_dim=mu.var(0).tolist(),
                                active_units=int((mu.var(0) > 0.01).sum()),
                                sensitivity_scale3=float(np.mean(sens)) if sens else None)
    for key in keys:
        out[f"fm_{key}"] = {"inside": (err[key][:, 0] / cnt[:, 0]).round(5).tolist(),
                            "outside": (err[key][:, 1] / cnt[:, 1]).round(5).tolist()}
    for key in keys[1:]:
        out[f"gap_{key}_inside_pct"] = (100 * (err[key][:, 0] / err["real"][:, 0] - 1)).round(3).tolist()
        out[f"gap_{key}_outside_pct"] = (100 * (err[key][:, 1] / err["real"][:, 1] - 1)).round(3).tolist()
    dst = OUTPUTS / "eval" / "action_use"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"{args.run}_{step}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
