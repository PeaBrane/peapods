import numpy as np
from peapods import Ising
from peapods.cli import build_parser, run_simulate


def test_heat_capacity_excludes_disorder_variance():
    kwargs = {
        "couplings": "bimodal",
        "temperatures": np.array([0.8, 1.6], dtype=np.float32),
        "n_disorder": 4,
        "seed": 3,
    }
    plain = Ising((4, 4), **kwargs)
    result = plain.sample(512, pt_interval=1)
    physics = Ising((4, 4), **kwargs)
    physics.sample(512, pt_interval=1, collect_physics=True)

    per_sample = physics.physics["per_disorder"]["heat_capacity"].mean(axis=0)
    np.testing.assert_allclose(plain.heat_capacity, per_sample, rtol=1e-6)
    pooled = 16 * (result["energies2"] - result["energies"] ** 2) / plain.temperatures**2
    assert np.all(pooled > plain.heat_capacity * 1.01)


def test_cli_table_prints_top_cluster_sizes(capsys):
    args = build_parser().parse_args(
        [
            "simulate",
            "--shape",
            "4",
            "4",
            "--couplings",
            "bimodal",
            "--temp-min",
            "1",
            "--temp-max",
            "2",
            "--n-temps",
            "2",
            "--n-sweeps",
            "8",
            "--n-replicas",
            "2",
            "--overlap-cluster-update-interval",
            "1",
            "--collect-cluster-stats",
            "--seed",
            "1",
        ]
    )
    run_simulate(args)
    assert "Top-4 Clusters" in capsys.readouterr().out
