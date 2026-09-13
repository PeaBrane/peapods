"""XY public contracts and short, seeded finite-system checks."""

import numpy as np
import pytest
from peapods import XY, Ising


def assert_results_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], dict):
            assert_results_equal(a[key], b[key])
        else:
            np.testing.assert_equal(a[key], b[key], err_msg=key)


def test_reset_and_parallel_replay_with_signed_disorder_and_tempering():
    options = dict(
        lattice_shape=(4, 4),
        couplings="gaussian",
        temperatures=[0.6, 1.2, 2.4],
        n_replicas=2,
        n_disorder=3,
        seed=785,
    )
    serial, parallel = XY(**options), XY(**options)
    initial = serial._sim.get_spins().copy()
    settings = dict(
        n_sweeps=128,
        pt_interval=1,
        pt_schedule="full_ladder",
        collect_blocks=True,
        block_size=16,
        autocorrelation_max_lag=8,
        overrelaxation_sweeps=1,
        displacements=[[1, 0]],
        vortices=True,
    )
    a = serial.sample(**settings, sequential=True)
    b = parallel.sample(**settings, sequential=False)
    assert_results_equal(a, b)
    np.testing.assert_equal(serial._sim.get_spins(), parallel._sim.get_spins())
    np.testing.assert_equal(
        serial._sim.get_system_ids(), parallel._sim.get_system_ids()
    )
    serial.reset()
    np.testing.assert_equal(initial, serial._sim.get_spins())
    assert_results_equal(a, serial.sample(**settings, sequential=True))
    serial.reset(seed=77)
    changed = serial._sim.get_spins().copy()
    serial.reset(seed=77)
    np.testing.assert_equal(changed, serial._sim.get_spins())
    assert not np.array_equal(initial, changed)
    assert np.all(a["per_disorder"]["parallel_tempering"]["edge_attempts"] == 256)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sweep_mode": "gibbs"},
        {"sweep_mode": "none", "cluster_update_interval": None},
        {"cluster_update_interval": 0},
        {"cluster_updates": 0},
        {"pt_interval": 0},
        {"warmup_ratio": float("nan")},
        {"warmup_ratio": -1},
        {"warmup_ratio": 1},
        {"collect_blocks": True, "block_size": 0},
        {"displacements": [[1]]},
        {"autocorrelation_backend": "fft"},
        {"overlap_cluster_update_interval": 1},
    ],
)
def test_invalid_options_fail_before_mutation(kwargs):
    model = XY((4, 4), temperatures=[1.0, 2.0], seed=91)
    before = model._sim.get_spins().copy()
    with pytest.raises((ValueError, TypeError)):
        model.sample(10, **kwargs)
    np.testing.assert_equal(model._sim.get_spins(), before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lattice_shape": (2, 4)},
        {"temperatures": [0]},
        {"temperatures": [np.inf]},
        {"couplings": np.full((4, 4, 2), np.nan)},
        {"occupation": np.ones((4, 4))},
        {"n_replicas": 0},
        {"n_disorder": 0},
        {"geometry": "triangular"},
    ],
)
def test_invalid_constructors(kwargs):
    options = dict(lattice_shape=(4, 4), temperatures=[1.0], seed=4)
    options.update(kwargs)
    with pytest.raises(ValueError):
        XY(**options)


def test_uncoupled_finite_volume_moments_masks_and_partial_blocks():
    mask = np.zeros((4, 4), dtype=bool)
    mask.flat[:7] = True
    model = XY(
        (4, 4),
        couplings=np.zeros((4, 4, 2)),
        temperatures=[1.0],
        occupation=mask,
        seed=82,
    )
    result = model.sample(
        32768,
        warmup_ratio=0,
        cluster_update_interval=None,
        displacements=[[0, 0], [1, 0]],
        collect_blocks=True,
        block_size=127,
        sequential=True,
    )
    n, occupied = 16, 7
    expected_m2 = occupied / n**2
    expected_m4 = (2 * occupied**2 - occupied) / n**4
    np.testing.assert_allclose(result["mags2"], expected_m2, rtol=0.02)
    np.testing.assert_allclose(result["mags4"], expected_m4, rtol=0.025)
    np.testing.assert_allclose(
        result["binder_cumulant"], 1 / (2 * occupied), atol=0.014
    )
    np.testing.assert_equal(result["energies"], [0.0])
    np.testing.assert_equal(result["heat_capacity"], [0.0])
    np.testing.assert_allclose(result["correlations"][0, 0], occupied / n, atol=1e-14)
    assert abs(result["correlations"][0, 1]) < 0.01
    blocks = result["per_disorder"]["blocks"]
    assert blocks["counts"].sum() == 32768
    assert blocks["counts"][0, -1] == 32768 % 127
    for name, sums in blocks["sums"].items():
        mean = sums.sum(axis=1) / blocks["counts"].sum(axis=1).reshape(
            (-1,) + (1,) * (sums.ndim - 2)
        )
        np.testing.assert_allclose(mean, result["per_disorder"][name], atol=1e-13)


def test_vacancies_zero_bonds_and_three_dimensions():
    empty = XY(
        (3, 3), temperatures=[0.5, 2.0], occupation=np.zeros((3, 3), dtype=bool), seed=8
    )
    assert np.all(empty.couplings == 0)
    result = empty.sample(8, vortices=True, displacements=[[0, 0]], sequential=True)
    for key in (
        "mags",
        "energies",
        "heat_capacity",
        "susceptibility",
        "intact_plaquette_fraction",
    ):
        assert np.all(result[key] == 0)
    assert np.all(np.isnan(result["binder_cumulant"]))
    assert np.all(np.isnan(result["correlation_length"]))
    assert np.all(np.isnan(result["angle_vortex_density"]))
    cube = XY((3, 3, 3), couplings="bimodal", temperatures=[1.0], seed=7)
    result = cube.sample(64, overrelaxation_sweeps=2)
    assert result["helicity_modulus"].shape == (1, 3)
    np.testing.assert_allclose(
        np.linalg.norm(cube._sim.get_spins(), axis=-1), 1, atol=1e-12
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        cube.sample(1, vortices=True)


def test_cluster_clock_and_anisotropic_energy():
    coupling = np.ones((4, 4, 2))
    coupling[..., 1] = -2.0
    model = XY((4, 4), couplings=coupling, temperatures=[0.05], n_replicas=2, seed=126)
    result = model.sample(
        8192,
        sweep_mode="none",
        cluster_updates=2,
        cluster_update_interval=2,
        sequential=True,
    )
    counts = result["per_disorder"]
    assert counts["cluster_updates"][0, 0] == 8192 * 2
    assert counts["visited_spins"][0, 0] == 8192 * 2 * 16
    assert -3.001 < result["energies"][0] < -2.94
    np.testing.assert_allclose(
        result["helicity_d"].sum(axis=-1) / 16, -result["energies"], atol=1e-12
    )


def test_opt_in_ising_physics_and_thermal_disorder_variance():
    couplings = np.stack([np.ones((4, 4, 2)), np.ones((4, 4, 2)) * 2])
    model = Ising((4, 4), couplings=couplings, temperatures=np.array([0.1]), seed=3)
    result = model.sample(
        256,
        cluster_update_interval=1,
        collect_physics=True,
        block_size=32,
        displacements=[[1, 0]],
        sequential=True,
    )
    physics = result["physics"]
    np.testing.assert_allclose(
        physics["per_disorder"]["energies"][:, 0], [-2.0, -4.0], atol=1e-12
    )
    assert physics["heat_capacity"][0] == 0.0
    assert model.heat_capacity[0] == 0.0
    assert "helicity_modulus" not in physics
    assert "physics" not in Ising((4, 4), temperatures=np.array([1.0]), seed=3).sample(
        2
    )


def test_signed_tempering_agrees_with_independent_temperatures():
    options = dict(
        lattice_shape=(4, 4),
        couplings="bimodal",
        temperatures=[0.8, 1.3, 2.0],
        seed=923,
    )
    independent = XY(**options).sample(16384, sequential=True)
    tempered = XY(**options).sample(
        16384, pt_interval=1, pt_schedule="full_ladder", sequential=True
    )
    np.testing.assert_allclose(
        independent["energies"], tempered["energies"], atol=0.025
    )
    np.testing.assert_allclose(independent["mags2"], tempered["mags2"], atol=0.012)


@pytest.mark.parametrize("model_class,dtype", [(Ising, np.float32), (XY, np.float64)])
def test_constructor_owns_input_arrays(model_class, dtype):
    temperatures = np.array([1.0, 2.0], dtype=dtype)
    couplings = np.ones((4, 4, 2), dtype=dtype)
    model = model_class((4, 4), temperatures=temperatures, couplings=couplings, seed=4)
    temperatures[:] = 9
    couplings[:] = -1
    np.testing.assert_equal(model.temperatures, [1.0, 2.0])
    np.testing.assert_equal(model.couplings, np.ones((4, 4, 2)))
