"""RD, forest-odds and NWB comparisons for every combined FD row.

usage: python rd_nwb.py fd_mc.csv conductance.csv out.csv
C_eff and NW are computed with unit conductances (fd-mc conductance) and scale by y = sinh^2 beta.
RD: Q/(1-Q) <= C_eff.  Forest theorem: P/(1-P) <= C_eff.  NWB: Q/(1-Q) <= NW (NW >= C_eff).
"""
import csv
import math
import sys

fd, cond, out = sys.argv[1:4]
C = {(r["lattice"], r["size"], r["pair"]): r for r in csv.DictReader(open(cond))}
rows = []
for r in csv.DictReader(open(fd)):
    key = (r["lattice"], r["size"], r["pair"])
    if key not in C:
        continue
    b = float(r["beta"])
    y = math.sinh(b) ** 2
    q, sq, p, sp = (float(r[k]) for k in ("Q", "SE_Q", "AG", "SE_AG"))
    ce, nw = y * float(C[key]["Ceff1"]), y * float(C[key]["NW1"])
    oq, soq = q / (1 - q), sq / (1 - q) ** 2
    op, sop = p / (1 - p), sp / (1 - p) ** 2
    rows.append(dict(
        lattice=r["lattice"], size=r["size"], pair=r["pair"], u=r["u"], v=r["v"], beta=b, dist=C[key]["dist"],
        Q=q, SE_Q=sq, AG=p, SE_AG=sp, FD_ratio=float(r["ratio"]), z_FD=float(r["z"]),
        Ceff=ce, NW=nw, odds_Q=oq, SE_odds_Q=soq,
        RD_ratio=oq / ce, z_RD=(oq - ce) / max(soq, 1e-300),
        odds_AG=op, SE_odds_AG=sop, forest_ratio=op / ce, z_forest=(op - ce) / max(sop, 1e-300),
        NWB_ratio=oq / nw, z_NWB=(oq - nw) / max(soq, 1e-300), note=r["note"],
        rse_FD=math.hypot(sq / q, sp / p) if q > 0 and p > 0 else float("inf")))
keys = list(rows[0].keys())
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
res = [r for r in rows if r["note"] != "unresolved"]
print(f"{len(rows)} rows ({len(res)} resolved)")
print("forest-theorem check: max P/(1-P) / C_eff = %.5f, max z_forest = %.2f" % (max(r["forest_ratio"] for r in res), max(r["z_forest"] for r in res)))
print("RD: max ratio %.5f, max z_RD %.2f; NWB: max ratio %.5f, max z_NWB %.2f" % (
    max(r["RD_ratio"] for r in res), max(r["z_RD"] for r in res), max(r["NWB_ratio"] for r in res), max(r["z_NWB"] for r in res)))
prec = [r for r in res if r["rse_FD"] <= 0.01]
print(f"\n{'lattice':<10}{'size':<11}{'pair':<14}{'beta':>5} {'FD':>8} {'RD':>8} {'NWB':>8} {'forest':>8}   (ratios; 15 tightest FD rows with SE(FD ratio) <= 1%)")
for r in sorted(prec, key=lambda r: -r["FD_ratio"])[:15]:
    print(f"{r['lattice']:<10}{r['size']:<11}{r['pair']:<14}{r['beta']:>5} {r['FD_ratio']:>8.4f} {r['RD_ratio']:>8.4f} {r['NWB_ratio']:>8.4f} {r['forest_ratio']:>8.4f}")
print("\nper beta (rows with SE(FD ratio) <= 1%): max FD ratio | max RD ratio | max NWB ratio")
for b in sorted({r["beta"] for r in prec}):
    rb = [r for r in prec if r["beta"] == b]
    print(f"  beta={b:<5} FD {max(r['FD_ratio'] for r in rb):.4f}  RD {max(r['RD_ratio'] for r in rb):.4f}  NWB {max(r['NWB_ratio'] for r in rb):.4f}")
