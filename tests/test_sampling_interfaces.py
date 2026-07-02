import numpy as np
import pytest
from peapods import Ising
from peapods.cli import _load_sweep_config, build_parser
from peapods.sweep import (
    _flatten_per_disorder_arrays,
    _run_child_seed,
    _run_seed_words,
)


def test_explicit_seed_controls_couplings_and_reset_replays_dynamics():
    temperatures = np.array([1.0, 2.0], dtype=np.float32)
    first = Ising(
        (4, 4),
        couplings="bimodal",
        temperatures=temperatures,
        n_replicas=2,
        seed=41,
    )
    second = Ising(
        (4, 4),
        couplings="bimodal",
        temperatures=temperatures,
        n_replicas=2,
        seed=41,
    )
    initial_spins = first._sim.get_spins().copy()

    np.testing.assert_array_equal(first.couplings, second.couplings)
    np.testing.assert_array_equal(initial_spins, second._sim.get_spins())

    first.sample(2, warmup_ratio=0)
    first.reset()
    np.testing.assert_array_equal(first._sim.get_spins(), initial_spins)

    first.reset(seed=99)
    seeded_reset = first._sim.get_spins().copy()
    first.reset(seed=99)
    np.testing.assert_array_equal(first._sim.get_spins(), seeded_reset)
    first.reset()
    np.testing.assert_array_equal(first._sim.get_spins(), initial_spins)


def test_disorder_zero_is_stable_when_disorder_count_grows():
    one = Ising((4, 4), couplings="gaussian", n_disorder=1, seed=7)
    many = Ising((4, 4), couplings="gaussian", n_disorder=3, seed=7)
    np.testing.assert_array_equal(one.couplings, many.couplings[0])


def test_fk_observe_shapes_and_noncanonical_winding_omission():
    model = Ising(
        (4, 4),
        temperatures=np.array([1.5, 2.5]),
        n_disorder=2,
        neighbor_offsets=[[1, 0], [0, 1]],
        seed=5,
    )
    result = model.sample(
        2,
        cluster_update_interval=1,
        cluster_mode="sw",
        cluster_action="observe",
        warmup_ratio=0,
    )
    observed = result["per_disorder"]["cluster_observations"]["fk"]

    assert observed["observation_count"].shape == (2, 2)
    assert observed["observation_count"].dtype == np.uint64
    assert observed["cluster_size_counts"].shape == (2, 2, 17)
    assert observed["top_four_component_fractions"].shape == (2, 2, 4)
    assert "winding_x" not in observed


def test_cmr_observe_and_full_ladder_pt_results():
    model = Ising(
        (4, 4),
        couplings="bimodal",
        temperatures=np.array([1.0, 2.0, 4.0]),
        n_replicas=2,
        seed=11,
    )
    result = model.sample(
        2,
        overlap_cluster_update_interval=1,
        overlap_cluster_build_mode="cmr",
        overlap_cluster_mode="sw",
        overlap_cluster_action="observe",
        pt_interval=1,
        pt_schedule="full_ladder",
        warmup_ratio=0,
    )
    per_disorder = result["per_disorder"]
    cmr = per_disorder["cluster_observations"]["cmr_blue"]
    pt = per_disorder["parallel_tempering"]

    assert cmr["observation_count"].shape == (1, 3)
    assert np.all(cmr["observation_count"] == 2)
    assert pt["edge_attempts"].shape == (1, 2)
    assert np.all(pt["edge_attempts"] == 4)
    assert pt["round_trips"].shape == (1, 2, 3)

    continued = model.sample(
        1,
        pt_interval=1,
        pt_schedule="full_ladder",
        warmup_ratio=0,
    )
    assert np.all(continued["per_disorder"]["parallel_tempering"]["edge_attempts"] == 6)

    model.reset()
    reset = model.sample(
        1,
        pt_interval=1,
        pt_schedule="full_ladder",
        warmup_ratio=0,
    )
    assert np.all(reset["per_disorder"]["parallel_tempering"]["edge_attempts"] == 2)


@pytest.mark.parametrize(
    ("build_mode", "result_key"),
    [("houdayer", "houdayer"), ("jorg", "jorg")],
)
def test_other_sw_overlap_observers_are_supported(build_mode, result_key):
    model = Ising(
        (4, 4),
        couplings="bimodal",
        temperatures=np.array([1.5]),
        n_replicas=2,
        seed=31,
    )
    result = model.sample(
        1,
        overlap_cluster_update_interval=1,
        overlap_cluster_build_mode=build_mode,
        overlap_cluster_mode="sw",
        overlap_cluster_action="observe",
        warmup_ratio=0,
    )
    observed = result["per_disorder"]["cluster_observations"][result_key]
    assert observed["observation_count"].tolist() == [[1]]


def test_unsupported_observe_fails_before_mutation():
    model = Ising((4, 4), temperatures=np.array([2.0]), seed=13)
    before = model._sim.get_spins().copy()
    with pytest.raises(ValueError, match="requires cluster_mode='sw'"):
        model.sample(
            1,
            cluster_update_interval=1,
            cluster_mode="wolff",
            cluster_action="observe",
            warmup_ratio=0,
        )
    np.testing.assert_array_equal(model._sim.get_spins(), before)


def test_autocorrelation_backend_defaults_and_fft_agree():
    model_kwargs = {
        "lattice_shape": (4, 4),
        "couplings": "bimodal",
        "temperatures": np.array([1.0, 2.0], dtype=np.float32),
        "n_replicas": 2,
        "seed": 37,
    }
    default = Ising(**model_kwargs).sample(
        64,
        autocorrelation_max_lag=8,
        warmup_ratio=0,
        sequential=True,
    )
    explicit_ring = Ising(**model_kwargs).sample(
        64,
        autocorrelation_max_lag=8,
        autocorrelation_backend="ring",
        warmup_ratio=0,
        sequential=True,
    )
    fft = Ising(**model_kwargs).sample(
        64,
        autocorrelation_max_lag=8,
        autocorrelation_backend="fft",
        warmup_ratio=0,
        sequential=True,
    )

    np.testing.assert_array_equal(default["mags2_tau"], explicit_ring["mags2_tau"])
    np.testing.assert_allclose(
        fft["mags2_tau"], default["mags2_tau"], rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        fft["overlap2_tau"], default["overlap2_tau"], rtol=0, atol=1e-9
    )


def test_invalid_autocorrelation_backend_fails_before_sampling():
    model = Ising((4, 4), temperatures=np.array([1.0, 2.0]), seed=43)
    before = model._sim.get_spins().copy()

    with pytest.raises(ValueError, match="must be 'ring' or 'fft'"):
        model.sample(4, autocorrelation_backend="other", warmup_ratio=0)
    with pytest.raises(ValueError, match="requires autocorrelation_max_lag"):
        model.sample(4, autocorrelation_backend="fft", warmup_ratio=0)

    np.testing.assert_array_equal(model._sim.get_spins(), before)


def test_cli_and_toml_propagate_v021_options(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "simulate",
            "--shape",
            "4",
            "4",
            "--temp-min",
            "1",
            "--temp-max",
            "2",
            "--n-sweeps",
            "2",
            "--seed",
            "17",
            "--cluster-action",
            "observe",
            "--pt-schedule",
            "full_ladder",
            "--overlap-cluster-action",
            "observe",
            "--autocorrelation-max-lag",
            "8",
            "--autocorrelation-backend",
            "fft",
        ]
    )
    assert args.seed == 17
    assert args.cluster_action == "observe"
    assert args.pt_schedule == "full_ladder"
    assert args.overlap_cluster_action == "observe"
    assert args.autocorrelation_backend == "fft"

    config = tmp_path / "sweep.toml"
    config.write_text(
        """
[sampling]
seed = 23
[cluster]
action = "observe"
[parallel_tempering]
schedule = "full_ladder"
[overlap_cluster]
action = "observe"
[diagnostics.autocorrelation]
max_lag = 8
backend = "fft"
"""
    )
    loaded = _load_sweep_config(config)
    assert loaded["seed"] == 23
    assert loaded["cluster_action"] == "observe"
    assert loaded["pt_schedule"] == "full_ladder"
    assert loaded["overlap_cluster_action"] == "observe"
    assert loaded["autocorrelation_max_lag"] == 8
    assert loaded["autocorrelation_backend"] == "fft"


def test_run_sweep_child_seed_and_npz_flattening_are_stable(tmp_path):
    words = _run_seed_words(29)
    expected = _run_child_seed(words, "bimodal", (4, 8))
    assert expected == _run_child_seed(_run_seed_words(29), "bimodal", (4, 8))
    assert expected != _run_child_seed(words, "gaussian", (4, 8))
    assert expected != _run_child_seed(words, "bimodal", (8, 4))

    per_disorder = {
        "cluster_observations": {
            "fk": {"observation_count": np.ones((1, 2), dtype=np.uint64)}
        },
        "parallel_tempering": {
            "edge_attempts": np.ones((1, 1), dtype=np.uint64),
            "edge_acceptances": np.zeros((1, 1), dtype=np.uint64),
            "round_trips": np.zeros((1, 2, 2), dtype=np.uint64),
        },
    }
    flat = _flatten_per_disorder_arrays(per_disorder, prefix="4x4")
    path = tmp_path / "result.npz"
    np.savez(path, **flat)
    with np.load(path, allow_pickle=False) as saved:
        assert "4x4_per_disorder_cluster_observations_fk_observation_count" in saved
        assert "4x4_per_disorder_pt_edge_attempts" in saved
        assert all(saved[key].dtype != object for key in saved.files)
