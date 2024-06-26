"""Test fanpy.wavefunction.wavefunctions."""
import functools

from fanpy.wfn.base import BaseWavefunction

import numpy as np

import pytest

from utils import disable_abstract, skip_init


def test_assign_nelec():
    """Test BaseWavefunction.assign_nelec."""
    test = skip_init(disable_abstract(BaseWavefunction))
    # check errors
    with pytest.raises(TypeError):
        test.assign_nelec(None)
    with pytest.raises(TypeError):
        test.assign_nelec(2.0)
    with pytest.raises(TypeError):
        test.assign_nelec("2")
    with pytest.raises(ValueError):
        test.assign_nelec(0)
    with pytest.raises(ValueError):
        test.assign_nelec(-2)
    # int
    test.assign_nelec(2)
    assert test.nelec == 2


def test_nspin():
    """Test BaseWavefunction.nspin."""
    test = skip_init(disable_abstract(BaseWavefunction))
    # check errors
    with pytest.raises(TypeError):
        test.assign_nspin(None)
    with pytest.raises(TypeError):
        test.assign_nspin(2.0)
    with pytest.raises(TypeError):
        test.assign_nspin("2")
    with pytest.raises(ValueError):
        test.assign_nspin(0)
    with pytest.raises(ValueError):
        test.assign_nspin(-2)
    with pytest.raises(ValueError):
        test.assign_nspin(3)
    # int
    test.assign_nspin(2)
    assert test.nspin == 2


def test_assign_memory():
    """Test BaseWavefunction.assign_memory."""
    test = skip_init(disable_abstract(BaseWavefunction))
    test.assign_memory(None)
    assert test.memory == np.inf
    test.assign_memory(10)
    assert test.memory == 10
    test.assign_memory(20.0)
    assert test.memory == 20
    test.assign_memory("10mb")
    assert test.memory == 1e6 * 10
    test.assign_memory("20.1gb")
    assert test.memory == 1e9 * 20.1
    with pytest.raises(TypeError):
        test.assign_memory([])
    with pytest.raises(ValueError):
        test.assign_memory("20.1kb")


def temp_assign_params(self, params=None, add_noise=False, default_params=np.identity(10)):
    """Assign the parameters of the wavefunction."""
    if params is None:
        params = default_params

    BaseWavefunction.assign_params(self, params=params, add_noise=add_noise)


def test_assign_params():
    """Test BaseWavefunction.assign_params."""
    with pytest.raises(NotImplementedError):
        skip_init(disable_abstract(BaseWavefunction)).assign_params(None)
    # default
    test = skip_init(
        disable_abstract(BaseWavefunction, dict_overwrite={"assign_params": temp_assign_params})
    )
    test.assign_params()
    assert np.allclose(test.params, np.identity(10))

    test = skip_init(
        disable_abstract(BaseWavefunction, dict_overwrite={"assign_params": temp_assign_params})
    )
    test.assign_params()
    assert np.allclose(test.params, np.identity(10))

    # check errors
    test = skip_init(
        disable_abstract(BaseWavefunction, dict_overwrite={"assign_params": temp_assign_params})
    )

    # check noise
    test = skip_init(
        disable_abstract(BaseWavefunction, dict_overwrite={"assign_params": temp_assign_params})
    )
    test.assign_params(add_noise=True)
    assert np.all(np.abs(test.params - np.identity(10)) <= 0.2 / 100)
    assert not np.allclose(test.params, np.identity(10))

    test = skip_init(
        disable_abstract(BaseWavefunction, dict_overwrite={"assign_params": temp_assign_params})
    )
    test.assign_params(params=np.identity(10).astype(complex), add_noise=True)
    assert np.all(np.abs(np.real(test.params - np.identity(10))) <= 0.1 / 100)
    assert np.all(np.abs(np.imag(test.params - np.identity(10))) <= 0.01 * 0.1 / 100)
    assert not np.allclose(np.real(test.params), np.identity(10))
    assert not np.allclose(np.imag(test.params), np.zeros((10, 10)))

    # for testing one line of code
    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "assign_params": lambda self, params=None, add_noise=False: temp_assign_params(
                    self, params, add_noise, np.zeros((1, 1, 1))
                )
            },
        )
    )
    test.assign_params()
    assert test.params.shape == (1, 1, 1)


def test_init():
    """Test BaseWavefunction.__init__."""
    test = skip_init(disable_abstract(BaseWavefunction))
    BaseWavefunction.__init__(test, 2, 10, memory=20)
    assert test.nelec == 2
    assert test.nspin == 10
    assert test.memory == 20


def test_olp():
    """Test BaseWavefunction._olp."""
    test = skip_init(disable_abstract(BaseWavefunction))
    with pytest.raises(NotImplementedError):
        test._olp(0b0101)


def test_olp_deriv():
    """Test BaseWavefunction._olp_deriv."""
    test = skip_init(disable_abstract(BaseWavefunction))
    with pytest.raises(NotImplementedError):
        test._olp_deriv(0b0101)


def test_enable_cache():
    """Test BaseWavefunction.enable_cache."""
    test = skip_init(disable_abstract(BaseWavefunction))
    test.memory = 16
    test.params = np.array([1, 2, 3])
    test.enable_cache(include_derivative=False)
    assert hasattr(test, "_cache_fns")
    assert test._cache_fns["overlap"].cache_info().maxsize == 2
    with pytest.raises(KeyError):
        test._cache_fns["overlap derivative"]

    test.memory = 16
    test.enable_cache(include_derivative=True)
    assert test._cache_fns["overlap"].cache_info().maxsize == 0
    assert test._cache_fns["overlap derivative"].cache_info().maxsize == 0

    test.memory = 4 * 8
    test.enable_cache(include_derivative=True)
    assert test._cache_fns["overlap"].cache_info().maxsize == 1
    assert test._cache_fns["overlap derivative"].cache_info().maxsize == 1

    test.memory = np.inf
    test.enable_cache()
    assert test._cache_fns["overlap"].cache_info().maxsize == 2 ** 30


def test_clear_cache():
    """Test BaseWavefunction.clear_cache."""
    test = skip_init(disable_abstract(BaseWavefunction))

    def olp(sd):
        """Overlap of wavefunction."""
        return 0.0

    olp = functools.lru_cache(2)(olp)
    test._cache_fns = {}
    test._cache_fns["overlap"] = olp
    with pytest.raises(KeyError):
        test.clear_cache("overlap derivative")

    def olp_deriv(sd, deriv):
        """Return the derivative of the overlap of wavefunction."""
        return 0.0

    olp_deriv = functools.lru_cache(2)(olp_deriv)
    test._cache_fns["overlap derivative"] = olp_deriv

    olp(2)
    olp(3)
    olp_deriv(2, 0)
    olp_deriv(3, 0)
    assert olp.cache_info().currsize == 2
    assert olp_deriv.cache_info().currsize == 2
    test.clear_cache("overlap")
    assert olp.cache_info().currsize == 0
    assert olp_deriv.cache_info().currsize == 2
    test.clear_cache()
    assert olp.cache_info().currsize == 0
    assert olp_deriv.cache_info().currsize == 0


def test_nspatial():
    """Test BaseWavefunction.nspatial."""
    test = skip_init(disable_abstract(BaseWavefunction))
    test.assign_nspin(10)
    assert test.nspatial == 5


def test_nparams():
    """Test BaseWavefunction.nparams."""
    test = skip_init(disable_abstract(BaseWavefunction))
    test.params = np.arange(100)
    assert test.nparams == 100


def test_spin():
    """Test BaseWavefunction.spin."""
    test = skip_init(disable_abstract(BaseWavefunction))
    assert test.spin is None


def test_seniority():
    """Test BaseWavefunction.seniority."""
    test = skip_init(disable_abstract(BaseWavefunction))
    assert test.seniority is None
