# More than a Hack 2027 — Phase 1: Ball Possession
## Technical document

### 1. The approach in one paragraph

For every frame we compute a score for each of the three states — **H** (home ball), **A** (away
ball), **D** (dead ball) — from simple, physically meaningful evidence. We then choose the single
sequence of states that has the highest total score, where **every change of state costs a fixed
penalty**. That last step (a Viterbi decoder, the classic algorithm for the best path through a
hidden Markov model) is what turns noisy frame-by-frame guesses into whole possessions. A brief
touch by the other team does not earn enough evidence to pay for two switches, so the possession
is not split. A clear spell of control by the other team does, and the switch is placed where the
evidence starts, not after a delay, because the decoder sees the whole half at once.

### 2. Why this rather than something else

- The ranking metric (spellF1) rewards getting **whole possessions** right and punishes extra
  ones. A per-frame classifier plus smoothing (the baseline) trades one error for the other
  through a single window size. A decoder with a switch cost addresses the real question —
  *is there enough evidence that possession changed?* — directly.
- We have no labelled data from other matches, and rules forbid labels on this one. So every
  constant is a quantity with a physical meaning (metres, seconds, m/s), set from football
  reasoning, not learned.
- The decoder works the same whether or not the ball is visible: when the ball is lost (41% of
  graded frames) it simply carries the last state unless other evidence argues otherwise.

### 3. Assets used

| asset | used | how |
|---|---|---|
| Tactical ATD (players + ball) | **yes** | ball–player distances, ball position, ball speed, player speeds, team shape |
| Tactical Smart Tagging | **yes** | set pieces as dead-ball evidence; team-tagged attacking phases as a weak team prior |
| Tactical homography | no | not needed: the ATD is already in pitch coordinates |
| Tactical video | no | out of scope for the time available |
| Broadcast (all) | no | not synchronised to the tactical timeline; no clear gain for the cost |

### 4. Evidence (per frame)

All distances use metres (105.3 × 68.0 m pitch).

**Ball track.** Ball gaps up to 12 frames (~0.5 s) are filled by straight-line interpolation.
Longer gaps are left empty.

**Control (H vs A)**, only where the ball is seen:
`lean = tanh((dA − dH) / 1.5 m)`, scaled by `reach = clip(1 − min(dH, dA) / 4 m, 0, 1)`,
where dH / dA are the distances from the ball to the nearest home / away player.
+lean is added to H and −lean to A. So a team gets credit only when one of its players is close
to the ball *and* clearly closer than any opponent; a ball in flight gives no evidence.

**Dead ball (D)**:
- ball more than 0.5 m outside a line: +3 per frame;
- average player speed (over ±2 s) below 1.6 m/s — players walk during stoppages:
  up to +1.5 per frame, and down to −1.5 when play is fast;
- ball standing still (< 0.6 m/s over ~1 s): +1;
- ball unseen beyond the gap-fill limit: +0.6 (weak — the ball is also lost in live play);
- Smart Tagging set piece: +1 for the 3 s before and after the tag start. The tag opens about
  3.5 s before the restart kick, measured on the first-half kickoff whose frame is published in
  DATA.md.

**Team shape**, only where the ball is not seen: `tanh(log(area_H / area_A) / 0.5)`, weight 0.05,
where *area* is the product of the x and y spread of a team's tracked players. The team with the
ball spreads out and the other team closes up. On frames where the control evidence is clear,
the sign of this cue agrees with it **76%** of the time. That check uses only our own algorithmic
pseudo-labels, never the ground truth. Shape changes slowly, so frames are not independent, and
the weight is kept small (≈ log-odds 1.15 spread over ~2 s).

**Team-tagged phases** (Build Up, Progression, Ball in Final Third, Creating Chances,
Attacking Transition, Ball in the Box): ±0.15 per frame for the tagged team. Weak on purpose:
phases are not possessions and the tagging is automatic.

### 5. The possession rules we implemented

- **Change of possession** costs 12 (≈ 0.5 s of clear control by the other team). A touch,
  deflection or lost aerial duel without follow-up control is not enough.
- **Loose and contested balls** give no evidence either way, so possession stays with the
  last team, as CHALLENGE.md asks.
- **Entering or leaving a stoppage** costs 30. A stoppage needs sustained evidence: ball out,
  play slowed down, a tagged restart.
- **Restarts**: the team after a stoppage is whichever team's evidence wins after play resumes.
  We do not try to infer who "owns" a dead ball, since the task asks for D.
- Each half is decoded on its own. Frames outside the two halves are written as X.

### 6. How the main constant was chosen

The switch cost decides how many possessions come out. We set it from football knowledge,
without the ground truth. A pro match has roughly 100–120 stoppages and 250–300 possession
spells once same-team restarts are counted. On the full match, without any labels, the
decoder finds:

| switch cost | possessions | stoppages |
|---|---|---|
| 8 | 310 | 114 |
| **12 (used)** | **261** | **114** |
| 20 | 208 | 114 |
| 30 | 175 | 114 |

114 stoppages, the same at every setting, is a realistic number for a match, which suggests
the dead-ball evidence is sound. Final output: H 141 spells (median 10.6 s), A 120 spells
(median 6.1 s), D 114 stoppages; time share H 33%, A 18%, D 49% of graded frames.

### 7. Use of the public ground truth

Used only to check that the pipeline is correct (frame alignment, file format) and to read what
the labels mean, as CHALLENGE.md invites. No constant was chosen by score on it. For the record,
on those 5 minutes the final file scores macroF1 0.823 and spellF1 0.368, against 0.621 / 0.326
for `baseline_smooth50`. EVALUATION.md is clear that this window cannot rank systems: the
spellF1 gap is inside its noise. The macroF1 gap is large and comes mostly from dead-ball
detection (F1 for D of ~0.83 against ~0.32).

### 8. Training data

None. No labelled data from this game or any other game was used. The only "learning" is the
76% agreement check in section 4, which compares two of our own algorithmic signals.

### 9. Limitations and what we would do next

- **Hidden ball.** When the ball is lost through a change of possession, the switch is placed
  using weak cues (team shape), so its timing can be off by up to the length of the gap. Next:
  use the video to detect the ball, or predict the ball path from player movement.
- **Duels and second balls.** Proximity cannot tell a controlled touch from a rebound. Next:
  use ball direction changes and speed after the touch, and the next touch by the same team.
- **Quick restarts.** Fast throw-ins and free kicks have little slow-down, so short stoppages
  can be missed or merged into play.
- **Goalkeeper holding the ball** looks like a still ball and can be read as dead.
- **Fixed constants.** With labelled tracking from other matches, the weights and costs could
  be learned properly (for example a CRF or a small sequence model) instead of set by hand.

### 10. Reproduction

```bash
pip install -r solution/requirements.txt      # numpy==2.5.3
python solution/possession.py --name <team_name>
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

Deterministic: the same input data gives the same file byte for byte
(md5 `7eccdbf536e04216c0bc2491567493cf`).
