"""Project wall time per (lattice, size, beta, side) to reach SE/value <= target on classes with value >= qmin.

usage: python project.py target target_far qmin threads pilot.csv [...]
`target` applies to adjacent classes (name contains "adj" or "d1"), `target_far` to the rest.
Each pilot row records secs (wall, at `threads` threads); time scales as (se / (target * value))^2.
"""
import csv
import sys
from collections import defaultdict

target, target_far, qmin, threads = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])


def is_near(name):
    return "adj" in name or "d1" in name


groups = defaultdict(list)
for f in sys.argv[5:]:
    for r in csv.DictReader(open(f)):
        secs = float(dict(kv.split("=") for kv in r["extra"].split(";") if "=" in kv)["secs"])
        groups[(r["side"], r["lattice"], r["size"], float(r["beta"]))].append((r["pair"], float(r["value"]), float(r["se"]), secs, float(r["tau"])))
tot = defaultdict(float)
print(f"{'side':<3} {'lattice':<11}{'size':<10}{'beta':>5} {'pilot_s':>8} {'proj_s':>9} {'tau_max':>7} binding")
for k in sorted(groups):
    rows = groups[k]
    secs = rows[0][3]
    need, bind = 0.0, ""
    for pair, v, se, _, _ in rows:
        if v >= qmin and se > 0:
            f = (se / ((target if is_near(pair) else target_far) * v)) ** 2
            if f > need:
                need, bind = f, pair
    proj = secs * need
    tot[(k[0], k[1], k[2])] += proj
    print(f"{k[0]:<3} {k[1]:<11}{k[2]:<10}{k[3]:>5} {secs:>8.1f} {proj:>9.0f} {max(r[4] for r in rows):>7.2f} {bind}")
print("totals (s at the pilot thread count):")
for k, v in sorted(tot.items()):
    print(f"  {k[0]:<3} {k[1]:<11}{k[2]:<10} {v:>9.0f} s = {v / 3600:.2f} h")
print(f"grand total {sum(tot.values()) / 3600:.2f} h at {threads} threads")
