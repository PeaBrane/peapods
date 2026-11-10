"""Compare fd-mc validation rows with the Z1 exact tables and recorded 10x10/12x12 values.

usage: python compare_ref.py <dir with ref_q.csv ref_ag.csv ref_qglass.csv> <Z1 scratch dir>
"""
import csv
import json
import math
import sys
from pathlib import Path

run, z1 = Path(sys.argv[1]), Path(sys.argv[2])


def rows(name):
    p = run / name
    return list(csv.DictReader(p.open())) if p.exists() else []


exact = {}
for f, size in [("outA/grid6x6.jsonl", "6x6"), ("outA2/grid7x7.jsonl", "7x7"), ("outA/box3x3x3.jsonl", "3x3x3")]:
    for line in (z1 / f).open():
        r = json.loads(line)
        exact[(size, f"{r['pair'][0]}-{r['pair'][1]}", round(r["beta"], 4))] = (r["Q"], r["AG"])
# 6x6 interior pairs double as the 6x6x1 box
for (size, pair, b), v in list(exact.items()):
    if size == "6x6":
        exact[("6x6x1", pair, b)] = v

recorded = {}
names = {"corner-adj": lambda L: "0-1", "boundary-adj": lambda L: "1-2", "row0-end-to-end": lambda L: f"0-{L - 1}"}
for line in (z1 / "t_mc_all.txt").open():
    p = line.replace("[", " ").replace("]", " ").replace(",", " ").split()
    if len(p) < 8 or not p[0].isdigit() or p[2] not in names:
        continue
    L, beta, q, lo, hi, ag = int(p[0]), float(p[1]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
    se = (hi - lo) / 2 / math.sqrt(2 * math.log(4 / 1e-3))
    recorded[(f"{L}x{L}", names[p[2]](L), round(beta, 4))] = (q, se, ag)

print(f"{'size':<7}{'pair':<7}{'beta':>5} {'side':<4}{'fd-mc':>22} {'reference':>22} {'z':>7}  ref")
worst = 0.0
nrows = 0
for side, files in [("Q", ["ref_q.csv", "ref_qglass.csv"]), ("AG", ["ref_ag.csv"])]:
    for f in files:
        for r in rows(f):
            key = (r["size"], r["pair"], round(float(r["beta"]), 4))
            v, se = float(r["value"]), float(r["se"])
            if side == "AG":  # one-event floor for rare connections
                se = max(se, 1.0 / (float(r["n"]) * float(r["nimages"])))
            if key in exact:
                ref, rse, kind = exact[key][0 if side == "Q" else 1], 0.0, "exact"
            elif key in recorded:
                q, qse, ag = recorded[key]
                ref, rse, kind = (q, qse, "Z1-MC") if side == "Q" else (ag, 0.0, "Z1-DP")
            else:
                continue
            z = (v - ref) / math.sqrt(se * se + rse * rse + 1e-30)
            worst = max(worst, abs(z))
            nrows += 1
            meth = "replica" if f == "ref_qglass.csv" else ("planar" if side == "Q" else "AG-MC")
            print(f"{key[0]:<7}{key[1]:<7}{key[2]:>5} {side:<4}{v:>12.6f}+-{se:<8.2e} {ref:>12.6f}+-{rse:<8.1e} {z:>7.2f}  {kind}/{meth}")
print(f"max |z| = {worst:.2f} over {nrows} comparisons")
