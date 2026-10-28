#!/usr/bin/env python
"""Train a surgical world model (EG-SWM or a baseline action interface) on SurgVU latent windows.

    python scripts/train/train_wm.py --name eg_swm_B --action arms --size B --steps 100000
    python scripts/train/train_wm.py --name mono_B --action global --size B --steps 100000

Outputs go to outputs/runs/<name>/: config.json, log.jsonl, ckpt_last.pt (+ milestones). --resume continues.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from swm.data.external import ExternalWindows
from swm.data.windows import LatentWindows
from swm.models.actions import region_from_boxes
from swm.models.world_model import ACTIONS, WMConfig, WorldModel
from swm.paths import OUTPUTS


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--action", choices=ACTIONS, default="arms")
    ap.add_argument("--size", choices=["S", "B", "L"], default="B")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--du", type=int, default=8)
    ap.add_argument("--dg", type=int, default=24)
    ap.add_argument("--kl", type=float, default=1e-4)
    ap.add_argument("--fwd", type=float, default=1.0, help="forward-decoder loss weight; 0 = diffusion loss only")
    ap.add_argument("--kl-warmup", type=int, default=10_000, help="steps over which the KL weight ramps from 0")
    ap.add_argument("--arm-pool", choices=["region", "global", "shuffled"], default="region")
    ap.add_argument("--scene", action="store_true", help="arms: add a scene action pooled outside the arm regions")
    ap.add_argument("--stabilise", action="store_true", help="infer learned actions from camera-stabilised latents")
    ap.add_argument("--p-drop-learned", type=float, default=0.0,
                    help="probability of dropping only the learned actions (camera kept) in training")
    ap.add_argument("--use-ui", action="store_true", help="console readout (hand, energy) in the arm tokens")
    ap.add_argument("--fp8", action="store_true", help="FP8 training of the transformer's linear layers (torchao)")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--train-fraction", type=float, default=1.0, help="fraction of training cases/episodes")
    ap.add_argument("--data", default="surgvu", help="surgvu or a prepared external dataset (suturebot, crcd, ...)")
    ap.add_argument("--stems", help="JSON {train: [...], val: [...]} of SurgVU parts; default all encoded parts")
    ap.add_argument("--init", help="initialise from outputs/runs/<run>/ckpt_last.pt (EMA weights, non-strict)")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def to_device(batch: dict) -> dict:
    out = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    out["region"] = region_from_boxes(out["boxes"], 14, 20, dilate=1)
    return out


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


@torch.no_grad()
def evaluate(model: WorldModel, loader: DataLoader, n_gen: int = 64) -> dict:
    """Context-mode flow-matching loss (2 clean context latents) with the inferred actions, and with actions taken
    from another window (all / learned only / camera only) -- the gap measures how much the actions explain;
    the forward decoder's error with own and shuffled actions; plus the latent error of action-replay
    generation."""
    model.eval()
    gen = torch.Generator(device="cuda").manual_seed(1234)
    sums = {"val_fm": 0.0, "val_fm_shuf_all": 0.0, "val_fm_shuf_learned": 0.0, "val_fm_shuf_camera": 0.0}
    n, gen_err, n_g = 0, 0.0, 0
    for batch in loader:
        b = to_device(batch)
        z = b["z"]
        bsz, t = z.shape[:2]
        tau = torch.rand(bsz, 1, device="cuda", generator=gen).expand(bsz, t).clone()
        tau[:, :2] = 0
        eps = torch.randn(z.shape, device="cuda", generator=gen)
        tt = tau[:, :, None, None, None]
        x = (1 - tt) * z + tt * eps
        ones = torch.ones(bsz, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            acts = model.infer_actions(z, b, stochastic=False)
            shuf = {k: (v.roll(1, 0) if k.startswith("u_") else v) for k, v in acts.items()}
            b_cam = dict(b, camera=b["camera"].roll(1, 0))
            for key, a, bb in [("val_fm", acts, b), ("val_fm_shuf_all", shuf, b_cam),
                               ("val_fm_shuf_learned", shuf, b), ("val_fm_shuf_camera", acts, b_cam)]:
                tok, mask = model.tokens(bb, a, ones)
                v = model.dit(x, model.cond(tau, bb, ones), tok, mask)
                sums[key] += (v.float() - (eps - z)).pow(2).mean(dim=(2, 3, 4))[:, 2:].sum().item()
            if model.fwd is not None:                    # are the actions informative at all?
                for key, a in (("val_fwd", acts), ("val_fwd_shuf", shuf)):
                    sums[key] = sums.get(key, 0.0) + model.forward_loss(z, b, a).item() * bsz * (t - 2)
        n += bsz * (t - 2)
        if n_g < n_gen:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(b, n_ctx=2, steps=10, generator=gen)
            gen_err += (out[:, 2:].float() - z[:, 2:]).pow(2).mean().item() * bsz
            n_g += bsz
    model.train()
    out = {k: v / max(n, 1) for k, v in sums.items()}
    out["val_gen_mse"] = gen_err / max(n_g, 1)
    return out


def main() -> None:
    args = parse()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    run = OUTPUTS / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    cfg = WMConfig(size=args.size, action=args.action, camera=not args.no_camera, du=args.du, dg=args.dg,
                   kl_weight=args.kl, fwd_weight=args.fwd, grad_checkpoint=args.grad_ckpt, arm_pool=args.arm_pool,
                   use_ui=args.use_ui, scene=args.scene, stabilise=args.stabilise,
                   p_drop_learned=args.p_drop_learned)
    torch.backends.cudnn.benchmark = True
    (run / "config.json").write_text(json.dumps({"model": cfg.to_dict(), "train": vars(args)}, indent=1))

    if args.data == "surgvu":
        train_cases = None
        if args.train_fraction < 1.0:
            sp = json.loads((Path(__file__).resolve().parents[2] / "configs/splits/surgvu_v1.json").read_text())
            g = torch.Generator().manual_seed(args.seed)
            n_keep = max(1, round(args.train_fraction * len(sp["train"])))
            keep = torch.randperm(len(sp["train"]), generator=g)[:n_keep]
            train_cases = [sp["train"][i] for i in keep.tolist()]
        stems = json.loads(Path(args.stems).read_text()) if args.stems else {}
        ds = LatentWindows("train", cases=train_cases, stems=stems.get("train"))
        val = LatentWindows("val", max_windows=512, seed=0, stems=stems.get("val"))
        (run / "parts.json").write_text(json.dumps({"train": [p["stem"] for p in ds.parts],
                                                    "val": [p["stem"] for p in val.parts]}, indent=1))
    else:
        kin = args.action == "kin"             # kinematics conditioning needs verified video-kinematics sync
        ds = ExternalWindows(args.data, "train", fraction=args.train_fraction, seed=args.seed, synced_only=kin)
        val = ExternalWindows(args.data, "val", synced_only=kin)
        if len(val) == 0:                      # no validation split: monitor on a fixed slice of training data
            val = ExternalWindows(args.data, "train", synced_only=kin)
            val.index = val.index[:: max(1, len(val.index) // 256)][:256]
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=args.bs, shuffle=False, num_workers=2)
    print(f"train windows {len(ds)}, val windows {len(val)}", flush=True)

    model = WorldModel(cfg).cuda()
    if args.init:
        st = torch.load(OUTPUTS / "runs" / args.init / "ckpt_last.pt", map_location="cuda", weights_only=False)
        src = WorldModel(WMConfig.from_dict(st["config"])).cuda()
        for p, e in zip(src.parameters(), st["ema"], strict=True):
            p.data.copy_(e)
        missing, unexpected = model.load_state_dict(src.state_dict(), strict=False)
        print(f"initialised from {args.init}: missing {sorted({k.split('.')[0] for k in missing})}, "
              f"unused {sorted({k.split('.')[0] for k in unexpected})}", flush=True)
        del src
    n_params = sum(p.numel() for p in model.parameters())
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "embed" in name.lower() or "null" in name or "offset" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.99), fused=True)
    ema = [p.detach().clone().float() for p in model.parameters()]
    step = 0
    ckpt = run / "ckpt_last.pt"
    if args.resume and ckpt.exists():
        st = torch.load(ckpt, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        for e, s in zip(ema, st["ema"], strict=True):
            e.copy_(s)
        step = st["step"]
        print(f"resumed at step {step}", flush=True)
    if args.fp8:                               # dynamic-scaled FP8 matmuls on Blackwell tensor cores
        from torchao.float8 import convert_to_float8_training

        def big_linear(mod, fqn):
            # token-heavy layers only: per-frame modulation and cross-attention keys over the few arm tokens
            # have row counts (B*T, B*T*5) that FP8 weight-gradient matmuls cannot take
            return (isinstance(mod, torch.nn.Linear) and "mod" not in fqn and "cross.kv" not in fqn
                    and mod.in_features % 16 == 0 and mod.out_features % 16 == 0
                    and min(mod.in_features, mod.out_features) >= 256)
        convert_to_float8_training(model.dit, module_filter_fn=big_linear)
    if not args.no_compile:
        model.dit.compile()
        if model.idm is not None:
            model.idm.compile()

    def save(path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ema": ema, "step": step,
                    "config": cfg.to_dict()}, tmp)
        tmp.replace(path)

    def swap_ema() -> None:
        for p, e in zip(model.parameters(), ema, strict=True):
            tmp = p.detach().clone()
            p.data.copy_(e)
            e.copy_(tmp)

    log = (run / "log.jsonl").open("a" if args.resume else "w")
    print(f"{args.name}: {n_params / 1e6:.1f}M params, action={cfg.action}, camera={cfg.camera}", flush=True)
    model.train()
    t0, seen = time.time(), 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            b = to_device(batch)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, logs = model.loss(b, kl_scale=min(1.0, step / max(1, args.kl_warmup)))
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                torch._foreach_lerp_(ema, [p.detach().float() for p in model.parameters()], 1 - args.ema)
            step += 1
            seen += args.bs
            if step % 50 == 0:
                dt = time.time() - t0
                rec = dict(step=step, lr=opt.param_groups[0]["lr"], grad_norm=float(gn), win_per_s=seen / dt,
                           mem_gb=torch.cuda.max_memory_allocated() / 1e9, **logs)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % 500 == 0:
                    print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}),
                          flush=True)
                t0, seen = time.time(), 0
            if step % args.eval_every == 0 or step == args.steps:
                swap_ema()
                ev = evaluate(model, val_loader)
                swap_ema()
                log.write(json.dumps({"step": step, **ev}) + "\n")
                log.flush()
                print(f"eval step {step}: {ev}", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save(ckpt)
                if step % (4 * args.save_every) == 0 or step == args.steps:
                    save(run / f"ckpt_{step:07d}.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
