"""Thermodynamic consistency of the spin-glass heat capacity.

For a disordered model the heat capacity is the disorder average of per-sample
energy fluctuations, c = N [<e^2> - <e>^2] / T^2, and it must integrate to the
change of the disorder-averaged energy: int c dT = u(T_max) - u(T_min). Pooling
the fluctuations across disorder samples adds N beta^2 Var_J(<e>_J), which breaks
the identity by far more than the statistical error at low temperature.
"""

import numpy as np
from peapods import Ising

L = 8
N_DISORDER = 128
N_SWEEPS = 1 << 14
TEMPS = np.round(np.linspace(0.5, 2.5, 41), 4).astype(np.float32)
SEED = 7


def jackknife(n, stat):
    full = stat(np.ones(n, bool))
    reps = []
    for d in range(n):
        keep = np.ones(n, bool)
        keep[d] = False
        reps.append(stat(keep))
    reps = np.array(reps)
    return full, np.sqrt((n - 1) * reps.var(axis=0))


def main():
    kw = dict(couplings="bimodal", temperatures=TEMPS, n_disorder=N_DISORDER, seed=SEED)
    sample = dict(pt_interval=1, warmup_ratio=0.25)
    plain = Ising((L, L), **kw)
    plain.sample(N_SWEEPS, **sample)
    phys = Ising((L, L), **kw)
    phys.sample(N_SWEEPS, collect_physics=True, **sample)

    t = TEMPS.astype(float)
    default = np.asarray(plain.heat_capacity, float)
    physics = np.asarray(phys.physics["heat_capacity"], float)
    rel = np.max(np.abs(default - physics) / physics)
    print(f"max |default - per-sample| / per-sample = {rel:.2e}")
    assert rel < 1e-5, "default heat capacity disagrees with per-sample fluctuations"

    per = phys.physics["per_disorder"]
    c_samples = per["heat_capacity"]
    u_samples = per["energies"]
    c_int, c_err = jackknife(
        N_DISORDER, lambda k: np.trapezoid(c_samples[k].mean(axis=0), t)
    )
    du, du_err = jackknife(
        N_DISORDER,
        lambda k: u_samples[k].mean(axis=0)[-1] - u_samples[k].mean(axis=0)[0],
    )
    # The two estimates share samples, so the combined error is conservative.
    z = (c_int - du) / np.hypot(c_err, du_err)
    print(f"int c dT = {c_int:.4f} +- {c_err:.4f}   du = {du:.4f} +- {du_err:.4f}")
    print(f"z = {z:.2f}")
    assert abs(z) < 4, f"int c dT deviates from du by {z:.1f} sigma"

    dudt, dudt_err = jackknife(
        N_DISORDER, lambda k: np.diff(u_samples[k].mean(axis=0)) / np.diff(t)
    )
    c_mid, c_mid_err = jackknife(
        N_DISORDER,
        lambda k: (
            0.5 * (c_samples[k].mean(axis=0)[1:] + c_samples[k].mean(axis=0)[:-1])
        ),
    )
    z_mid = (c_mid - dudt) / np.hypot(c_mid_err, dudt_err)
    print(
        f"max |z| of c(T) against du/dT over {len(z_mid)} points: {np.abs(z_mid).max():.2f}"
    )
    assert np.abs(z_mid).max() < 5, "c(T) disagrees with du/dT"
    print("PASSED")


if __name__ == "__main__":
    main()
