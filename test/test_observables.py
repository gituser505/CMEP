import pytest
import numpy as np
from pathlib import Path

from mylib.observables import (
    magnetization,
    energy,
    jackknife_samples,
    jackknife,
    get_observable_arrays,
    get_observables
)


def test_magnetization_boundary_states():
    """Test if magnetization = 1.0/0.0 for magnetized/thermal states.
    """
    L = 4
    spins_up = np.ones((2, L, L), dtype=np.int8)
    spins_down = np.zeros((2, L, L), dtype=np.int8)
    
    checkerboard = np.indices((L, L)).sum(axis=0) % 2
    checkerboard = np.array([checkerboard, checkerboard])

    np.testing.assert_allclose(magnetization(spins_up), 1.0)
    np.testing.assert_allclose(magnetization(spins_down), 1.0)
    np.testing.assert_allclose(magnetization(checkerboard), 0.0)


def test_energy_ferromagnetic_ground_states():
    """Test if max/min energy per spin = +/-2.0 * J.
    """
    L = 4
    J = 1.0
    spins_up = np.ones((2, L, L), dtype=np.int8)
    calculated_energies = energy(spins_up, J=J)
    np.testing.assert_allclose(calculated_energies, -2.0 * J)

    checkerboard = np.indices((L, L)).sum(axis=0) % 2
    checkerboard = np.array([checkerboard, checkerboard])
    calculated_energies = energy(checkerboard, J=J)
    np.testing.assert_allclose(calculated_energies, 2.0*J)


def test_jackknife_samples_computation():
    """Test if Numba jackknife calculations correctly compute leave-one-out metrics.
    """
    data = np.array([2.0, 4.0, 4.0, 2.0], dtype=np.float64)
    
    jk_mean, jk_var, jk_bc = jackknife_samples(data)
    
    # Hand-calculate first index (leave out 0)
    expected_m1 = (4.0 + 4.0 + 2.0) / 3.0  
    expected_m2 = (16.0 + 16.0 + 4.0) / 3.0 
    expected_variance = expected_m2 - (expected_m1 ** 2)
    
    np.testing.assert_approx_equal(jk_mean[0], expected_m1)
    np.testing.assert_approx_equal(jk_var[0], expected_variance)
    assert len(jk_mean) == 4


def test_jackknife_standard_errors_on_flat_data():
    """test if jackknife calculates standard errors and handles flat data correctly.
    """
    flat_data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    mean_err, var_err, bc_err = jackknife(flat_data)
    
    assert mean_err == 0.0
    assert var_err == 0.0
    assert bc_err == 0.0


def test_get_observable_arrays():
    """Test if data configurations parse and filter cleanly into correct beta dictionaries.
    """
    spins = np.ones((3, 4, 4), dtype=np.int8)
    betas = np.array([0.2, 0.5, 0.2])
    M_dict, E_dict = get_observable_arrays(spins, betas)
    
    assert list(M_dict.keys()) == [0.2, 0.5]
    assert len(M_dict[0.2]) == 2
    assert len(M_dict[0.5]) == 1


def test_get_observables_thermodynamics():
    """Test nested structure formats output matrices correctly with correct sizes.
    """
    M_arrays = {0.2: np.array([0.1, 0.2, 0.15]), 0.4: np.array([0.8, 0.85, 0.9])}
    E_arrays = {0.2: np.array([-0.5, -0.6, -0.55]), 0.4: np.array([-1.8, -1.85, -1.9])}

    obs = get_observables(M_arrays, E_arrays, L=4)
    
    for observable in ['M', 'E', 'chi', 'C', 'bc']:
        assert observable in obs
        assert isinstance(obs[observable]['val'], np.ndarray)
        assert isinstance(obs[observable]['err'], np.ndarray)
        assert len(obs[observable]['val']) == 2
        assert len(obs[observable]['err']) == 2

