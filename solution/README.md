# Ball possession — Viterbi over tracking + Smart Tagging

Our entry for **More than a Hack 2027 — Phase 1: Ball Possession**.
The approach, the assets used and the limits are in [TECHNICAL.md](TECHNICAL.md).

## Reproduce the submission

Python 3.9+ (tested on 3.13). numpy only.

```bash
pip install -r solution/requirements.txt
# put the match data in ./data/tactical/ (see DATA.md), or set MTAH_DATA_DIR
python solution/possession.py --name <team_name>
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

Files read: `BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`,
`BAR-ATH_tactical_smart_tagging.xml`. Nothing else (no video, no homography, no broadcast).

The run takes a few seconds, has no randomness, and produces the same file byte for byte
(md5 `7eccdbf536e04216c0bc2491567493cf`). The first run writes a cache
`data/tactical/atd_cache.npz`; deleting it changes nothing in the output.

The match data is **not** in this repository and must not be added (`data/*` is gitignored).

## Files

| file | what |
|---|---|
| `solution/load.py` | reads the raw ATD into numpy arrays |
| `solution/possession.py` | evidence, Viterbi decoding, CSV writer; all constants in `PARAMS` |
| `solution/TECHNICAL.md` | the technical document |
