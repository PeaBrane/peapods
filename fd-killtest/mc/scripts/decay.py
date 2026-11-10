"""Boundary decay from a corner: Q(r) and P^AG(r), exponential-rate and power-law fits per beta.

usage: python decay.py q_decay.csv ag_decay.csv[,ag_more.csv] points.csv fits.csv [rmin]
Duplicate AG points take the stronger estimator (est=star2 over star).
Writes fits.csv (common window where both sides are resolved) and fits_own.csv (each side on its own
resolved window).
Fit window: r >= rmin and both sides resolved (value > 0, SE/value < 0.25; AG from the star estimator
also needs >= 30 events, double-star AG points only the SE criterion).
Weighted least squares on ln(value) with weights (value/SE)^2; parameter SEs are inflated by
sqrt(chi2/dof) when that exceeds 1. Points share samples, so SEs are approximate.
"""
import csv
import math
import sys
from collections import defaultdict

qf, af, pts_out, fit_out = sys.argv[1:5]
rmin = int(sys.argv[5]) if len(sys.argv) > 5 else 4
side = defaultdict(dict)
rank = {}
for f, s in [(qf, "Q")] + [(x, "AG") for x in af.split(",")]:
    for r in csv.DictReader(open(f)):
        rr = int(r["pair"].split("-r")[1])
        k = (r["lattice"], r["size"], float(r["beta"]))
        rk = 2 if "est=star2" in r["extra"] else 1
        if rank.get((k, s, rr), 0) > rk:
            continue
        rank[(k, s, rr)] = rk
        # double-star rows: the SE criterion alone decides resolution (hit counts do not apply)
        hits = float("inf") if rk == 2 and s == "AG" else float(r["n"]) * float(r["nimages"])
        side[k][(s, rr)] = (float(r["value"]), float(r["se"]), hits)


def wls(xs, ys, ws):
    sw = sum(ws); sx = sum(w * x for w, x in zip(ws, xs)); sy = sum(w * y for w, y in zip(ws, ys))
    sxx = sum(w * x * x for w, x in zip(ws, xs)); sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    d = sw * sxx - sx * sx
    b = (sw * sxy - sx * sy) / d
    a = (sy - b * sx) / sw
    chi2 = sum(w * (y - a - b * x) ** 2 for w, x, y in zip(ws, xs, ys))
    dof = max(len(xs) - 2, 1)
    se_b = math.sqrt(sw / d) * max(1.0, math.sqrt(chi2 / dof))
    return b, se_b, chi2 / dof


with open(pts_out, "w") as fp, open(fit_out, "w") as ff, open(fit_out.replace(".csv", "_own.csv"), "w") as fo:
    fo.write("lattice,size,beta,side,rmin,rmax,kappa,se_kappa,chi2_exp,eta,se_eta,chi2_pow\n")
    fp.write("lattice,size,beta,r,Q,SE_Q,AG,SE_AG,ratio,in_fit\n")
    ff.write("lattice,size,beta,rmin,rmax,npts,kappa_Q,se_kappa_Q,chi2_exp_Q,kappa_AG,se_kappa_AG,chi2_exp_AG,gap_kappa,se_gap_kappa,"
             "eta_Q,se_eta_Q,chi2_pow_Q,eta_AG,se_eta_AG,chi2_pow_AG,gap_eta,se_gap_eta\n")
    print(f"{'lattice':<11}{'size':<9}{'beta':>5} {'r-range':>9} {'kappa_Q':>15} {'kappa_AG':>15} {'gap':>15} | {'eta_Q':>13} {'eta_AG':>13} {'gap':>13}  chi2r(exp Q/AG, pow Q/AG)")
    for key in sorted(side):
        d = side[key]
        rs = sorted({rr for (_, rr) in d})
        use = []
        for rr in rs:
            if ("Q", rr) not in d or ("AG", rr) not in d:
                continue
            q, sq, _ = d[("Q", rr)]
            p, sp, n = d[("AG", rr)]
            ok = rr >= rmin and q > 0 and p > 0 and sq / q < 0.25 and sp / p < 0.25 and p * n >= 30
            fp.write(f"{key[0]},{key[1]},{key[2]},{rr},{q:.6e},{sq:.3e},{p:.6e},{sp:.3e},{(q / p if p > 0 else float('nan')):.5f},{int(ok)}\n")
            if ok:
                use.append((rr, q, sq, p, sp))
        own = {}
        for sd in ("Q", "AG"):
            pts = []
            for rr in rs:
                if (sd, rr) not in d or rr < rmin:
                    continue
                v, se, n = d[(sd, rr)]
                if v > 0 and se / v < 0.25 and (sd == "Q" or v * n >= 30):
                    pts.append((rr, v, se))
            if len(pts) >= 3:
                kb = wls([u[0] for u in pts], [math.log(u[1]) for u in pts], [(u[1] / u[2]) ** 2 for u in pts])
                eb = wls([math.log(u[0]) for u in pts], [math.log(u[1]) for u in pts], [(u[1] / u[2]) ** 2 for u in pts])
                own[sd] = (pts[0][0], pts[-1][0], -kb[0], kb[1], kb[2], -eb[0], eb[1], eb[2])
        for sd, o in own.items():
            fo.write(f"{key[0]},{key[1]},{key[2]},{sd},{o[0]},{o[1]},{o[2]:.5f},{o[3]:.5f},{o[4]:.2f},{o[5]:.4f},{o[6]:.4f},{o[7]:.2f}\n")
        if len(use) < 3:
            continue
        out = []
        for xf in (lambda r: r, lambda r: math.log(r)):
            for k in (1, 3):
                xs = [xf(u[0]) for u in use]
                ys = [math.log(u[k]) for u in use]
                ws = [(u[k] / u[k + 1]) ** 2 for u in use]
                b, se, c2 = wls(xs, ys, ws)
                out.append((-b, se, c2))
        (kq, skq, cq), (ka, ska, ca), (eq, seq, pq), (ea, sea, pa) = out
        gk, sgk = kq - ka, math.hypot(skq, ska)
        ge, sge = eq - ea, math.hypot(seq, sea)
        ff.write(f"{key[0]},{key[1]},{key[2]},{use[0][0]},{use[-1][0]},{len(use)},{kq:.5f},{skq:.5f},{cq:.2f},{ka:.5f},{ska:.5f},{ca:.2f},{gk:.5f},{sgk:.5f},"
                 f"{eq:.4f},{seq:.4f},{pq:.2f},{ea:.4f},{sea:.4f},{pa:.2f},{ge:.4f},{sge:.4f}\n")
        print(f"{key[0]:<11}{key[1]:<9}{key[2]:>5} {use[0][0]:>4}-{use[-1][0]:<4} {kq:>7.4f}+-{skq:<6.4f} {ka:>7.4f}+-{ska:<6.4f} {gk:>7.4f}+-{sgk:<6.4f} | "
              f"{eq:>6.3f}+-{seq:<5.3f} {ea:>6.3f}+-{sea:<5.3f} {ge:>6.3f}+-{sge:<5.3f}  {cq:.1f}/{ca:.1f}, {pq:.1f}/{pa:.1f}")
