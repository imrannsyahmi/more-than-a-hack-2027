"""Ball possession from automatic tracking + Smart Tagging, decoded with a 3-state Viterbi.

States: H (home ball, alive), A (away ball, alive), D (dead ball).

Every frame gets a score for each state from three kinds of evidence:
  * control  -- which team is closest to the ball (when the ball is seen)
  * deadness -- ball out of the pitch, ball standing still, players slowing down, ball lost for long
  * tagging  -- Smart Tagging set pieces (restarts) and team-tagged phases of play
A path through the states is then chosen to maximise total score minus a fixed cost for every
change of state. The change cost is what turns "who is nearest the ball this frame" into whole
possessions: a brief touch by the other team is not enough evidence to pay for two switches.

All constants are set from football reasoning (distances in metres, times in seconds) and fixed
before scoring. None were fitted to the public ground truth.

    python solution/possession.py            # writes evaluation/results/<name>.csv
"""
import argparse
import warnings
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
from load import load_atd  # noqa: E402
import load_smart_tagging  # noqa: E402

FPS = 25
L, W = GAMES[GAME]["pitch"]
H_, A_, D_ = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

PARAMS = dict(
    # --- ball track -------------------------------------------------------------------------
    gap_fill_frames=12,        # interpolate the ball across gaps up to ~0.5 s
    # --- control evidence -------------------------------------------------------------------
    control_scale_m=1.5,       # metres: how much closer one team must be to count as "on the ball"
    control_reach_m=4.0,       # metres: beyond this nobody is really on the ball
    w_control=1.0,
    # --- dead-ball evidence -----------------------------------------------------------------
    oob_margin_m=0.5,          # ball this far outside a line is out of play
    w_oob=3.0,
    slow_speed=1.6,            # m/s: mean player speed below this looks like a stoppage
    w_slow=1.5,
    speed_window_s=2.0,
    still_ball_speed=0.6,      # m/s: ball not moving
    still_window_s=1.0,
    w_still=1.0,
    w_missing_long=0.6,        # ball unseen for longer than gap_fill (per frame)
    # --- smart tagging ----------------------------------------------------------------------
    # A tag opens ~3.5 s before the restart kick (measured on the first-half kickoff, whose
    # frame is published in DATA.md), so the stoppage is expected around the tag start.
    setpiece_before_s=3.0,
    setpiece_after_s=3.0,
    w_setpiece=1.0,
    w_phase=0.15,              # weak prior from team-tagged phases (BUILD UP, PROGRESSION, ...)
    # --- team shape (only where the ball is not seen) ---------------------------------------
    # The team in possession spreads out, the other one closes up. On frames where the ball
    # control evidence is clear, sign(log area_H/area_A) agrees with it 76% of the time
    # (algorithmic pseudo-labels, no ground truth). Shape changes slowly, so successive frames
    # are far from independent: log-odds ~1.15 spread over ~2 s gives ~0.02-0.05 per frame.
    w_shape=0.05,
    # --- switching costs --------------------------------------------------------------------
    cost_team_switch=12.0,     # ~0.5 s of clear control by the other team
    cost_dead=30.0,            # entering or leaving a stoppage
)

PHASE_CODES_ATTACK = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
                      "ATTACKING TRANSITION", "BALL IN THE BOX"}


def fill_gaps(x, max_gap):
    """Linear interpolation of NaN runs no longer than max_gap, per column."""
    x = x.copy()
    n = len(x)
    ok = ~np.isnan(x[:, 0])
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return x, ok
    gaps = np.diff(idx)
    for k in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
        a, b = idx[k], idx[k + 1]
        t = (np.arange(a + 1, b) - a) / (b - a)
        x[a + 1:b] = x[a] + (x[b] - x[a]) * t[:, None]
    return x, ~np.isnan(x[:, 0])


def rolling_mean(v, half):
    """Centred nan-aware mean over +/- half frames."""
    ok = ~np.isnan(v)
    c = np.concatenate([[0], np.cumsum(np.where(ok, v, 0.0))])
    k = np.concatenate([[0], np.cumsum(ok)])
    n = len(v)
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    cnt = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / cnt, np.nan)


def evidence(p=PARAMS):
    P, S, B, _, team = load_atd()
    n = len(B)
    Pm = P * np.array([L, W], np.float32)
    Bm, seen = fill_gaps(B * np.array([L, W], np.float32), p["gap_fill_frames"])

    score = np.zeros((n, 3), np.float32)

    # ---- control: nearest home vs nearest away player ----------------------------------------
    d = np.linalg.norm(Pm - Bm[:, None, :], axis=2)  # (n, slots), NaN where either is missing
    home = team == 0
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dH = np.nanmin(np.where(home[None], d, np.nan), axis=1)
        dA = np.nanmin(np.where(~home[None], d, np.nan), axis=1)
    dH = np.where(np.isnan(dH), 99.0, dH)
    dA = np.where(np.isnan(dA), 99.0, dA)
    near = np.minimum(dH, dA)
    lean = np.tanh((dA - dH) / p["control_scale_m"])          # +1 home closer, -1 away closer
    reach = np.clip(1.0 - near / p["control_reach_m"], 0, 1)  # 0 when nobody is near the ball
    ctrl = np.where(seen, p["w_control"] * lean * reach, 0.0)
    score[:, H_] += ctrl
    score[:, A_] -= ctrl

    # ---- dead ball --------------------------------------------------------------------------
    m = p["oob_margin_m"]
    with np.errstate(invalid="ignore"):
        oob = seen & ((Bm[:, 0] < -m) | (Bm[:, 0] > L + m) | (Bm[:, 1] < -m) | (Bm[:, 1] > W + m))
    score[:, D_] += p["w_oob"] * oob

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vbar = np.nanmean(S, axis=1)
    vbar = rolling_mean(vbar, int(p["speed_window_s"] * FPS))
    slow = np.clip((p["slow_speed"] - np.nan_to_num(vbar, nan=p["slow_speed"])) / 0.5, -1, 1)
    score[:, D_] += p["w_slow"] * slow

    # ball speed from positions (m/s), raw observations only
    raw_ok = ~np.isnan(B[:, 0])
    Braw = B * np.array([L, W], np.float32)
    v = np.full(n, np.nan)
    both = raw_ok[2:] & raw_ok[:-2]
    v[1:-1][both] = np.linalg.norm(Braw[2:][both] - Braw[:-2][both], axis=1) * FPS / 2
    vb = rolling_mean(v, int(p["still_window_s"] * FPS / 2))
    still = raw_ok & (vb < p["still_ball_speed"])
    score[:, D_] += p["w_still"] * still

    score[:, D_] += p["w_missing_long"] * (~seen)

    # ---- team shape -------------------------------------------------------------------------
    def area(mask):
        X = np.where(mask[None, :, None], Pm, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanstd(X[:, :, 0], axis=1) * np.nanstd(X[:, :, 1], axis=1)
    with np.errstate(all="ignore"):
        cue = np.log(np.clip(area(home), 1.0, None) / np.clip(area(~home), 1.0, None))
    shape = np.where(~seen & ~np.isnan(cue), p["w_shape"] * np.tanh(np.nan_to_num(cue) / 0.5), 0.0)
    score[:, H_] += shape
    score[:, A_] -= shape

    # ---- smart tagging ----------------------------------------------------------------------
    for e in load_smart_tagging.load():
        a, b = e["start_frame"], e["end_frame"]
        if e["code"] == "SET PIECES":
            lo = max(0, a - int(p["setpiece_before_s"] * FPS))
            hi = min(n, a + int(p["setpiece_after_s"] * FPS))
            score[lo:hi, D_] += p["w_setpiece"]
        elif e["code"] in PHASE_CODES_ATTACK:
            t = e["tags"].get("Team", "N/A")
            k = H_ if t == "Barcelona" else A_ if t.startswith("At") else None
            if k is not None:
                score[a:b, k] += p["w_phase"]
                score[a:b, 1 - k] -= p["w_phase"]
    return score


def viterbi(score, cost):
    """Max-score path; cost[i, j] is paid when moving from state i to state j."""
    n, k = score.shape
    back = np.zeros((n, k), np.int8)
    acc = score[0].astype(np.float64).copy()
    for t in range(1, n):
        cand = acc[:, None] - cost          # from i to j
        back[t] = np.argmax(cand, axis=0)
        acc = cand[back[t], np.arange(k)] + score[t]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(acc))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(p=PARAMS):
    score = evidence(p)
    c = np.array([[0, p["cost_team_switch"], p["cost_dead"]],
                  [p["cost_team_switch"], 0, p["cost_dead"]],
                  [p["cost_dead"], p["cost_dead"], 0]], np.float64)
    n = source_frames()
    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        path = viterbi(score[first:last + 1], c)
        labels[first:last + 1] = LABELS[path]
    return labels


def write(labels, path):
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        for i, l in enumerate(labels):
            fh.write("{},{}\n".format(i, l))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="submission")
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = ap.parse_args()
    labels = predict()
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, args.name + ".csv")
    write(labels, out)
    u, c = np.unique(labels, return_counts=True)
    print(dict(zip(u, c)), "->", out)


if __name__ == "__main__":
    main()
