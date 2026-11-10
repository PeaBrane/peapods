"""Summarize the combined fd-mc CSV: per lattice coverage, tightest ratio, max z, top rows, candidates.

usage: python summarize.py fd_mc.csv
"Resolved" rows have at least 100 effective AG hits (AG x measurements x images) and |Q| >= 1e-10.
"""
import csv
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    for k in ("Q", "SE_Q", "AG", "SE_AG", "ratio", "z", "beta"):
        r[k] = float(r[k])
    r["se_ratio"] = (r["ratio"] * ((r["SE_Q"] / r["Q"]) ** 2 + (r["SE_AG"] / r["AG"]) ** 2) ** 0.5) if r["Q"] > 0 and r["AG"] > 0 else float("nan")
res = [r for r in rows if r["note"] != "unresolved"]
by = defaultdict(list)
for r in rows:
    by[(r["lattice"].split("#")[0], r["size"])].append(r)
print(f"{len(rows)} rows, {len(res)} resolved")
print(f"{'lattice':<11}{'size':<11}{'rows':>5}{'resolv':>7}{'max_ratio':>10}{'at':>28}{'min_ratio':>10}{'max_z':>7}{'max_z_all':>10}{'med_seR%':>9}")
for k in sorted(by):
    rs = by[k]
    rr = [r for r in rs if r["note"] != "unresolved"]
    if not rr:
        continue
    top = max(rr, key=lambda r: r["ratio"])
    sers = sorted(100 * r["se_ratio"] / r["ratio"] for r in rr if r["se_ratio"] == r["se_ratio"] and r["ratio"] > 0)
    print(f"{k[0]:<11}{k[1]:<11}{len(rs):>5}{len(rr):>7}{top['ratio']:>10.5f}{top['pair'] + ' b=' + str(top['beta']):>28}"
          f"{min(r['ratio'] for r in rr):>10.5f}{max(r['z'] for r in rr):>7.2f}{max(r['z'] for r in rs):>10.2f}{(sers[len(sers) // 2] if sers else float("nan")):>9.3f}")
def line(r):
    return (f"  {r['lattice']:<9}{r['size']:<11}{r['pair']:<18}b={r['beta']:<5} Q={r['Q']:.6f}+-{r['SE_Q']:.1e} AG={r['AG']:.6f}+-{r['SE_AG']:.1e} "
            f"ratio={r['ratio']:.5f}+-{r['se_ratio']:.5f} z={r['z']:.2f} tau={r['tau_int']}")


prec = [r for r in res if r["ratio"] > 0 and r["se_ratio"] / r["ratio"] <= 0.01]
print(f"\ntop 10 tightest precise rows (SE(ratio)/ratio <= 1%, {len(prec)} rows):")
for r in sorted(prec, key=lambda r: -r["ratio"])[:10]:
    print(line(r))
print("\ntop 10 rows by z (all resolved rows):")
for r in sorted(res, key=lambda r: -r["z"])[:10]:
    print(line(r))
print("\ntop 5 tightest resolved rows with SE(ratio) > 1% (rare-event far pairs):")
for r in sorted([r for r in res if r not in prec], key=lambda r: -r["ratio"])[:5]:
    print(line(r))
print("\nrows with z > 3 (watch list; candidates need z > 5):")
for r in sorted(rows, key=lambda r: -r["z"]):
    if r["z"] <= 3:
        break
    print(f"  {r['lattice']:<9}{r['size']:<11}{r['pair']:<18}b={r['beta']:<5} Q={r['Q']:.4e} AG={r['AG']:.4e} ratio={r['ratio']:.4f} z={r['z']:.2f} {r['note']}")
