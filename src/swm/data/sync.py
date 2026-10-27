"""Video-kinematics synchronisation of the external datasets.

The LeRobot conversions pair video frame i with kinematics row i, but in some episodes the two streams are
offset by a constant (CRCD episodes 12-14: kinematics lead by 1.2 s) or drift apart (CRCD episode 6: 0.6 s per
minute). We estimate the lag per episode by cross-correlating a video motion signal that does not involve the
endoscope ego-motion estimator (mean absolute change of the VAE latents between latent steps) with total arm
motion (normalised |d state| summed over dimensions), over the whole episode and per 2-minute chunk.
An episode is synced when the chunk lags agree with each other and with the global lag; its kinematics are then
shifted by that lag. Episodes that drift or never lock are excluded from every kinematics-based analysis.
Episodes shorter than one chunk (SutureBot: ~10 s) cannot be checked one by one: their correlation curves are
averaged, and all of them take the pooled lag if most episodes peak near it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WIN = 33
CHUNK = 1200            # 2 min at 10 fps
GLOBAL_LAG = 300        # search +-30 s over the whole episode
CHUNK_LAG = 100         # +-10 s per chunk
TOL = 5                 # chunk lags within +-0.5 s of the episode lag
MAX_LAG = 30            # larger constant offsets are treated as unreliable
SHORT_LAG = 20          # +-2 s for short episodes
MIN_SHIFT = 4           # lags below one latent step (0.4 s) are not resolvable here and are not applied


def episode_series(rows: pd.DataFrame, values: np.ndarray) -> dict[int, np.ndarray]:
    """Per-window arrays (N, 33, ...) -> per-episode 10 fps series (NaN where no window covers a frame)."""
    out = {}
    for ep, g in rows.groupby("episode"):
        idx, starts = g.index.to_numpy(), g["start"].to_numpy()
        s = np.full((starts.max() + WIN,) + values.shape[2:], np.nan, np.float32)
        for w, st in zip(idx, starts, strict=True):
            s[st:st + WIN] = values[w]
        out[int(ep)] = s
    return out


def _corr(a: np.ndarray, b: np.ndarray, lag: int, lo: int = 0, hi: int | None = None, min_n: int = 200) -> float:
    """corr(a[t + lag], b[t]) over t in [lo, hi): a positive lag means b (kinematics) leads a (video)."""
    hi = len(a) if hi is None else hi
    t = np.arange(max(lo, -lag), min(hi, len(a) - lag, len(b)))
    x, y = a[t + lag], b[t]
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > min_n and x[m].std() > 0 and y[m].std() > 0 else np.nan


def _curve(a, b, span, lo=0, hi=None, min_n=200) -> np.ndarray:
    return np.array([_corr(a, b, lag, lo, hi, min_n) for lag in range(-span, span + 1)])


def _best(a, b, span, lo=0, hi=None, min_n=200) -> tuple[int, float]:
    cs = _curve(a, b, span, lo, hi, min_n)
    if not np.isfinite(cs).any():
        return 0, np.nan
    k = int(np.nanargmax(cs))
    return k - span, float(cs[k])


def motion_signals(rows: pd.DataFrame, latents: np.ndarray, state: np.ndarray) -> tuple[dict, dict]:
    """Per-episode video motion (latent change, spread over the 4 frames of each latent step) and arm motion."""
    vid = np.full((len(rows), WIN), np.nan, np.float32)
    for w in range(len(rows)):
        z = np.asarray(latents[w], np.float32)
        vid[w, 1:] = np.repeat(np.abs(np.diff(z, axis=0)).mean(axis=(1, 2, 3)), 4)
    d = np.abs(np.diff(state, axis=1))
    d = d / (np.nanstd(d.reshape(-1, d.shape[-1]), axis=0) + 1e-9)
    kin = np.full((len(rows), WIN), np.nan, np.float32)
    kin[:, 1:] = np.nansum(d, axis=-1)
    v, k = episode_series(rows, vid), episode_series(rows, kin)
    k = {e: pd.Series(s).rolling(4, center=True, min_periods=1).mean().to_numpy() for e, s in k.items()}
    return v, k


def estimate(rows: pd.DataFrame, latents: np.ndarray, state: np.ndarray) -> dict[int, dict]:
    vid, kin = motion_signals(rows, latents, state)
    out = {}
    short = [ep for ep in vid if len(vid[ep]) < CHUNK]
    if short:
        curves = np.array([_curve(vid[ep], kin[ep], SHORT_LAG, min_n=30) for ep in short])
        pooled = np.nanmean(curves, axis=0)
        lag = int(np.nanargmax(pooled)) - SHORT_LAG
        own = [int(np.nanargmax(c)) - SHORT_LAG if np.isfinite(c).any() else None for c in curves]
        agree = float(np.mean([o is not None and abs(o - lag) <= TOL for o in own]))
        ok = bool(pooled[lag + SHORT_LAG] >= 0.2 and agree >= 0.7)
        for ep, o, c in zip(short, own, curves, strict=True):
            out[ep] = dict(lag=lag, ok=ok, pooled=True, pooled_corr=float(pooled[lag + SHORT_LAG]),
                           pooled_agreement=agree, own_lag=o, corr_at_0=float(c[SHORT_LAG]),
                           minutes=len(vid[ep]) / 600)
    for ep in vid:
        if ep in out:
            continue
        a, b = vid[ep], kin[ep]
        g_lag, g_corr = _best(a, b, GLOBAL_LAG)
        chunks = [_best(a, b, CHUNK_LAG, lo, lo + CHUNK)[0] for lo in range(0, len(a) - CHUNK, CHUNK)]
        lag = int(np.median(chunks)) if chunks else g_lag
        agree = float(np.mean([abs(c - lag) <= TOL for c in chunks])) if chunks else 0.0
        ok = (len(chunks) >= 2 and agree >= 0.7 and abs(g_lag - lag) <= TOL and abs(lag) <= MAX_LAG
              and np.isfinite(g_corr) and g_corr >= 0.2)
        out[ep] = dict(lag=lag, ok=bool(ok), pooled=False, global_lag=g_lag, global_corr=g_corr,
                       corr_at_0=_corr(a, b, 0),
                       chunk_lags=chunks, chunk_agreement=agree, minutes=len(a) / 600)
    for v in out.values():
        v["shift"] = v["lag"] if abs(v["lag"]) >= MIN_SHIFT else 0
    return out


def apply(rows: pd.DataFrame, signals: dict[str, np.ndarray], sync: dict[int, dict]) -> tuple[dict, np.ndarray]:
    """Shift every per-frame signal of a synced episode by its (resolvable) lag. Returns (signals, synced[N]);
    windows of unsynced episodes keep their original signals, windows whose shifted signals leave the episode are
    unsynced."""
    synced = np.zeros(len(rows), bool)
    out = {k: v.copy() for k, v in signals.items()}
    series = {k: episode_series(rows, v) for k, v in signals.items()}
    covered = episode_series(rows, np.ones((len(rows), WIN), np.float32))     # frames some window holds
    for w, (ep, start) in enumerate(zip(rows["episode"].to_numpy(), rows["start"].to_numpy(), strict=True)):
        s = sync[int(ep)]
        t0 = start - s["shift"]                # video frame f matches kinematics sample f - lag
        cov = covered[int(ep)]
        if not s["ok"] or t0 < 0 or t0 + WIN > len(cov) or not np.isfinite(cov[t0:t0 + WIN]).all():
            continue
        for k in signals:
            out[k][w] = series[k][int(ep)][t0:t0 + WIN]
        synced[w] = True
    return out, synced
