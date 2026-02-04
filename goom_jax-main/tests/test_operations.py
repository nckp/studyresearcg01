import jax
import jax.numpy as jnp
import pytest

from goom.goom import generate_random_gooms
from goom.operations import (
    log_add_exp,
    log_sum_exp,
    log_mean_exp,
    log_cum_sum_exp,
    log_cum_mean_exp,
    log_negate_exp,
    scale,
    scaled_exp,
    log_triu_exp,
    log_tril_exp,
    log_rmsnorm_exp,
    log_unitsquash_exp,
)


# Helper function to check for NaNs
def _check_no_nans(x):
    assert not jnp.isnan(x.real).any()
    assert not jnp.isnan(x.imag).any()


# Fixture for random goom tensors
@pytest.fixture
def random_gooms_2d():
    key = jax.random.PRNGKey(0)
    shape = (10, 5)
    gooms = generate_random_gooms(key, shape)
    _check_no_nans(gooms)
    return gooms


@pytest.fixture
def random_gooms_pair():
    key = jax.random.PRNGKey(1)
    shape = (10, 5)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    gooms1 = generate_random_gooms(subkey1, shape)
    gooms2 = generate_random_gooms(subkey2, shape)
    _check_no_nans(gooms1)
    _check_no_nans(gooms2)
    return gooms1, gooms2


# Tests for each function
def test_log_add_exp(random_gooms_pair):
    gooms1, gooms2 = random_gooms_pair
    result = log_add_exp(gooms1, gooms2)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_log_sum_exp(random_gooms_2d, axis):
    result = log_sum_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_log_mean_exp(random_gooms_2d, axis):
    result = log_mean_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_log_cum_sum_exp(random_gooms_2d, axis):
    result = log_cum_sum_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_log_cum_mean_exp(random_gooms_2d, axis):
    result = log_cum_mean_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


def test_log_negate_exp(random_gooms_2d):
    result = log_negate_exp(random_gooms_2d)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_scale(random_gooms_2d, axis):
    result = scale(random_gooms_2d, axis=axis)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_scaled_exp(random_gooms_2d, axis):
    result = scaled_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


@pytest.mark.parametrize("offset", [0, 1, -1])
def test_log_triu_exp(random_gooms_2d, offset):
    result = log_triu_exp(random_gooms_2d, diagonal_offset=offset)
    _check_no_nans(result)


@pytest.mark.parametrize("offset", [0, 1, -1])
def test_log_tril_exp(random_gooms_2d, offset):
    result = log_tril_exp(random_gooms_2d, diagonal_offset=offset)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_log_rmsnorm_exp(random_gooms_2d, axis):
    result = log_rmsnorm_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


def test_log_unitsquash_exp(random_gooms_2d):
    result = log_unitsquash_exp(random_gooms_2d)
    _check_no_nans(result)


# Tests for custom gradient functions

from goom.goom import goom_abs, goom_exp, goom_log


def test_goom_abs_matches_jnp_abs():
    values = jnp.linspace(-5.0, 5.0, 21, dtype=jnp.float32)
    assert jnp.allclose(goom_abs(values), jnp.abs(values))


def test_goom_abs_gradient_matches_spec():
    grad_fn = jax.grad(goom_abs)
    for x in (-2.0, -0.5, 0.0, 0.5, 3.0):
        x_arr = jnp.array(x, dtype=jnp.float32)
        grad = grad_fn(x_arr)
        expected = 1.0 if x == 0 else jnp.sign(x_arr)
        assert jnp.allclose(grad, expected)


def test_stable_exp_matches_jnp_exp():
    values = jnp.linspace(-5.0, 5.0, 11)
    assert jnp.allclose(goom_exp(values), jnp.exp(values))


def test_custom_gradient_matches_spec():
    grad_fn = jax.grad(goom_exp)
    for x in (-3.0, -0.1, 0.0, 0.5, 5.0):
        x_arr = jnp.array(x, dtype=jnp.float32)
        grad = grad_fn(x_arr)
        eps = jnp.finfo(x_arr.dtype).eps
        expected = x_arr + (eps if x >= 0 else -eps)
        assert jnp.allclose(grad, expected)


def test_goom_log_matches_jnp_log_for_positive_values():
    values = jnp.linspace(0.1, 10.0, 25, dtype=jnp.float32)
    assert jnp.allclose(goom_log(values), jnp.log(values))


def test_goom_log_gradient_matches_spec():
    grad_fn = jax.grad(goom_log)
    for x in (0.25, 1.0, 5.0, 50.0):
        x_arr = jnp.array(x, dtype=jnp.float32)
        grad = grad_fn(x_arr)
        eps = jnp.finfo(x_arr.dtype).eps
        expected = 1.0 / (x_arr + eps)
        assert jnp.allclose(grad, expected)
