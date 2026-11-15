"""Load the tactical ATD into numpy arrays (cached as .npz next to the data)."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import require, source_frames, GAME, GAMES  # noqa
from baseline import team_of_each_track  # noqa


def load_atd(cache=True):
    meta, raw = require(["atd_meta", "atd_raw"])
    cache_path = os.path.join(os.path.dirname(raw), "atd_cache.npz")
    team = team_of_each_track(meta)
    if cache and os.path.isfile(cache_path):
        z = np.load(cache_path)
        return z["P"], z["S"], z["B"], z["BS"], team
    n = source_frames()
    ns = len(team)
    P = np.full((n, ns, 2), np.nan, np.float32)
    S = np.full((n, ns), np.nan, np.float32)
    B = np.full((n, 2), np.nan, np.float32)
    BS = np.full(n, np.nan, np.float32)
    with open(raw) as fh:
        for line in fh:
            f, pl, ball = line.rstrip("\n").split(":")
            i = int(f) - 1
            vals = np.array(pl.replace(";", ",").split(","), dtype=np.float32).reshape(ns, 3)
            P[i] = vals[:, :2]
            S[i] = vals[:, 2]
            b = np.array(ball.split(","), dtype=np.float32)
            B[i] = b[:2]
            BS[i] = b[2]
    if cache:
        np.savez(cache_path, P=P, S=S, B=B, BS=BS)
    return P, S, B, BS, team
