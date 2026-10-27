#!/usr/bin/env python
"""Does the explicit camera action measure laparoscope motion annotated by surgeons? AutoLaparo Task 2.

AutoLaparo (Wang et al., MICCAI 2022) has 300 clips of human laparoscopic hysterectomy (10 s, 25 fps, 1080p, the
laparoscope held by an assistant), each labelled with the laparoscope motion that begins at T = 5 s and lasts
under a second: static, up, down, left, right, zoom-in or zoom-out. The original benchmark *predicts* that motion
from the frames before T; here the unchanged ego-motion estimator *measures* it from the frames after T. This
tests the camera action on another platform (Olympus laparoscope), procedure and annotation source.

Every clip is resampled to 10 fps, centre-cropped to the model aspect and converted to 320x224 grayscale exactly
like the external datasets, and every transition is estimated as in training (unreliable when fewer than 12
tracks survive or the RANSAC inlier ratio is below 0.5). Per clip the motion is the sum of the reliable
transitions in the window [5.0, 6.0) s. Reported:
  * detection of a labelled move (any non-static label) from the motion magnitude, ROC AUC (nothing fitted);
  * 7-class accuracy of a rule with five fitted numbers (static threshold, zoom weight and the sign convention of
    each axis), fitted on the official training clips (001-170) and scored on validation (171-227) and test
    (228-300); the direction accuracy on moving test clips; the confusion matrix;
  * sign agreement per axis on test clips; motion after T vs the second before it (moving clips); the mean
    motion magnitude over time for moving and static clips; sensitivity to the window.

    python scripts/eval/validate_camera_autolaparo.py [--workers 6]
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import av
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from swm.data import egomotion
from swm.data.geometry import MODEL_H, MODEL_W
from swm.paths import DATA, OUTPUTS, PROCESSED

SRC = DATA / "external" / "autolaparo" / "task2_laparoscope_motion"
CLASSES = ("static", "up", "down", "left", "right", "zoom_in", "zoom_out")
SPLITS = {"train": (1, 170), "val": (171, 227), "test": (228, 300)}
FPS, N = 10, 100                      # 10 s at 10 fps
MIN_INLIER = 0.5
WINDOW = (5.0, 6.0)                   # the labelled motion begins at 5 s and lasts < 1 s


def gray(img: np.ndarray) -> np.ndarray:
    """Centre crop to the model aspect, grayscale, 320x224 (as scripts/data/prepare_lerobot.py)."""
    ih, iw = img.shape[:2]
    if iw / ih > MODEL_W / MODEL_H:
        cw = int(round(ih * MODEL_W / MODEL_H))
        img = img[:, (iw - cw) // 2:(iw - cw) // 2 + cw]
    else:
        ch = int(round(iw * MODEL_H / MODEL_W))
        img = img[(ih - ch) // 2:(ih - ch) // 2 + ch]
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)


def clip_job(clip: int) -> tuple[np.ndarray, np.ndarray]:
    """Ego-motion of the 99 transitions of one clip at 10 fps; NaN where unreliable."""
    cv2.setNumThreads(1)
    with av.open(str(SRC / "clips" / f"{clip:03d}.mp4")) as c:
        fps = float(c.streams.video[0].average_rate)
        keep = {int(k * fps / FPS + 0.5): k for k in range(N)}
        frames = {}
        for i, f in enumerate(c.decode(video=0)):
            if i in keep:
                frames[keep[i]] = gray(f.to_ndarray(format="rgb24"))
    ego, inl = np.full((N, 4), np.nan, np.float32), np.zeros(N, np.float32)
    for k in range(1, N):
        if k in frames and k - 1 in frames:
            ego[k], inl[k], _ = egomotion.estimate(frames[k - 1], frames[k])
    ego[inl < MIN_INLIER] = np.nan
    return ego, inl


def estimate_all(workers: int) -> dict:
    cache = PROCESSED / "autolaparo" / "egomotion.npz"
    if cache.exists():
        return dict(np.load(cache))
    lab = np.loadtxt(SRC / "laparoscope_motion_label.txt", skiprows=1, dtype=int)
    with Pool(workers) as pool:
        res = pool.map(clip_job, lab[:, 0].tolist(), chunksize=4)
    out = dict(clip=lab[:, 0], label=lab[:, 1], phase=lab[:, 2],
               ego=np.stack([r[0] for r in res]), inlier=np.stack([r[1] for r in res]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def window_sum(ego: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Sum of reliable transitions whose end lies in (t0, t1]; transition k spans (k-1)/10 .. k/10 s."""
    k = np.arange(N) / FPS
    sel = (k > t0) & (k <= t1 + 1e-9)
    return np.nansum(ego[:, sel, :3], axis=1)                               # (clips, dx dy log_s)


def fit_rule(s: np.ndarray, y: np.ndarray) -> dict:
    """Sign convention per axis, zoom weight and static threshold that maximise training accuracy."""
    signs = {}
    for axis, (neg, pos) in {0: (3, 4), 1: (1, 2), 2: (6, 5)}.items():      # dx: left/right, dy: up/down, s: out/in
        a, b = s[y == neg, axis].mean(), s[y == pos, axis].mean()
        signs[axis] = 1.0 if b > a else -1.0                                # +1: positive value means class `pos`
    best = None
    for wz in np.logspace(-1, 1, 21):
        for thr in np.quantile(score(s, wz), np.linspace(0.02, 0.98, 97)):
            acc = (predict(s, dict(signs=signs, wz=wz, thr=thr)) == y).mean()
            if best is None or acc > best[0]:
                best = (acc, wz, thr)
    return dict(signs=signs, wz=float(best[1]), thr=float(best[2]), train_accuracy=float(best[0]))


def score(s: np.ndarray, wz: float = 1.0) -> np.ndarray:
    return np.abs(s[:, 0]) + np.abs(s[:, 1]) + wz * np.abs(s[:, 2])


def predict(s: np.ndarray, rule: dict) -> np.ndarray:
    w = np.abs(s) * np.array([1.0, 1.0, rule["wz"]])
    axis = w.argmax(1)
    pos = np.sign(s[np.arange(len(s)), axis]) * np.array([rule["signs"][a] for a in axis]) > 0
    cls = np.select([axis == 0, axis == 1], [np.where(pos, 4, 3), np.where(pos, 2, 1)], np.where(pos, 5, 6))
    return np.where(score(s, rule["wz"]) < rule["thr"], 0, cls)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    d = estimate_all(args.workers)
    ego, y, clip = d["ego"], d["label"], d["clip"]
    split = np.select([clip <= 170, clip <= 227], ["train", "val"], "test")
    s = window_sum(ego, *WINDOW)
    moving = y > 0
    out = {"clips": len(y), "window_s": WINDOW, "class_counts": np.bincount(y, minlength=7).tolist(),
           "reliable_transition_fraction": float(np.isfinite(ego[:, 1:, 0]).mean()),
           "reliable_fraction_in_window": float(np.isfinite(ego[:, 51:61, 0]).mean())}
    out["auc_moving_vs_static"] = {"all": float(roc_auc_score(moving, score(s)))}
    for name in SPLITS:
        m = split == name
        out["auc_moving_vs_static"][name] = float(roc_auc_score(moving[m], score(s[m])))
    pre = window_sum(ego, 3.0, 4.0)                                         # 1 s before the labelled motion
    out["auc_window_vs_before_moving_clips"] = float(roc_auc_score(
        np.r_[np.ones(moving.sum()), np.zeros(moving.sum())], np.r_[score(s[moving]), score(pre[moving])]))
    rule = fit_rule(s[split == "train"], y[split == "train"])
    out["rule"] = dict(rule, signs={["dx", "dy", "log_s"][a]: v for a, v in rule["signs"].items()})
    pred = predict(s, rule)
    for name in ("val", "test"):
        m = split == name
        mm = m & moving
        out[f"{name}_accuracy_7class"] = float((pred[m] == y[m]).mean())
        out[f"{name}_balanced_accuracy_7class"] = float(np.mean([(pred[m & (y == c)] == c).mean() for c in range(7)]))
        out[f"{name}_direction_accuracy_moving"] = float((predict(s[mm], dict(rule, thr=-1)) == y[mm]).mean())
        out[f"{name}_majority_baseline"] = float(np.bincount(y[m]).max() / m.sum())
    out["test_confusion"] = {"classes": CLASSES,
                             "matrix": confusion_matrix(y[split == "test"], pred[split == "test"],
                                                        labels=range(7)).tolist()}
    agree = {}
    for axis, (neg, pos) in {0: (3, 4), 1: (1, 2), 2: (6, 5)}.items():
        m = np.isin(y, (neg, pos)) & (split == "test")
        want = np.where(y[m] == pos, 1, -1) * rule["signs"][axis]
        agree[["dx", "dy", "log_s"][axis]] = float((np.sign(s[m, axis]) == want).mean())
    out["test_sign_agreement"] = agree
    mag = np.nan_to_num(np.abs(ego[..., 0]) + np.abs(ego[..., 1]) + np.abs(ego[..., 2]))
    out["profile_time_s"] = (np.arange(1, N) / FPS).round(1).tolist()
    out["profile_moving"] = mag[moving, 1:].mean(0).round(5).tolist()
    out["profile_static"] = mag[~moving, 1:].mean(0).round(5).tolist()
    out["sensitivity_window"] = {}
    for w in ((4.5, 5.5), (5.0, 5.5), (5.0, 6.0), (5.0, 6.5), (4.5, 6.5)):
        sw = window_sum(ego, *w)
        r = fit_rule(sw[split == "train"], y[split == "train"])
        out["sensitivity_window"][f"{w[0]}-{w[1]}"] = dict(
            auc=float(roc_auc_score(moving, score(sw))),
            test_accuracy_7class=float((predict(sw, r)[split == "test"] == y[split == "test"]).mean()))
    dst = OUTPUTS / "eval" / "camera_validation"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "autolaparo.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("profile")}, indent=1))


if __name__ == "__main__":
    main()
