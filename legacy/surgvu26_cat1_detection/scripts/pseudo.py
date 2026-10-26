"""Presence-constrained pseudo-labelling of the 280 training videos.

Weak supervision available per frame: which tool is installed on each robot arm
(USM1..USM4) from tools.csv.  We convert that into box-level labels by solving a
constrained assignment between detected boxes and installed instruments.

Why this works for the 6 classes that never appear in the public val set:
  if a frame has installed {needle driver, needle driver, tip-up grasper} and the
  classifier is confident about the two needle drivers, the remaining box *must*
  be the tip-up grasper.  Elimination bootstraps classes the model has never seen.
"""
import os, sys, json, pickle, itertools, numpy as np
sys.path.insert(0,os.path.dirname(__file__)); from common import NAMES, mask_ui
ROOT='/mnt/5385e3d3-fb4b-4846-8f72-55998fc781fb/surgical_Image/work'

def adjust(logp, prior, tau=0.5):
    """log p(c|x) - tau*log p_train(c); keeps rare classes competitive."""
    import numpy as _np
    pr=_np.asarray(prior,dtype=_np.float64); pr=pr/max(pr.sum(),1e-9)
    return logp - tau*_np.log(_np.clip(pr,1e-6,None))[None,:]

def assign(det_logp, installed_cls, installed_arm, box_cx, lam_spatial=0.35):
    """det_logp: (n,14) log-probs. installed_*: length-k arrays. box_cx: (n,) in [0,1].
    Returns (labels, score, margin) for the best injective matching of ALL n boxes
    onto distinct installed arms (requires n<=k), or None."""
    n=len(box_cx); k=len(installed_cls)
    if n==0 or n>k: return None
    # spatial prior: arms are usually installed left->right by USM index
    order=np.argsort(installed_arm)
    rank=np.empty(k); rank[order]=np.arange(k)
    exp_cx=(rank+0.5)/k
    scores=[]
    for perm in itertools.permutations(range(k), n):
        s=0.0
        for j,a in enumerate(perm):
            s+=det_logp[j, installed_cls[a]] - lam_spatial*abs(box_cx[j]-exp_cx[a])
        scores.append((s, perm))
    scores.sort(reverse=True)
    best_s,best_p=scores[0]
    # margin vs best alternative that yields a DIFFERENT label vector
    lbl=[installed_cls[a] for a in best_p]
    margin=np.inf
    for s,p in scores[1:]:
        if [installed_cls[a] for a in p]!=lbl:
            margin=best_s-s; break
    return np.array(lbl), best_s, margin

def frame_installed(armstate_row):
    """-> (classes, arms) of arms holding a real instrument; None if frame unusable."""
    cs=np.asarray(armstate_row)
    if (cs==-4).any() or (cs==-2).any(): return None   # conflicting / unknown tool
    idx=np.where(cs>=0)[0]
    if len(idx)==0: return None
    return cs[idx].astype(int), idx.astype(int)
