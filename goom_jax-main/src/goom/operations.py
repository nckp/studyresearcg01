import jax
import jax.numpy as jnp
import math

from goom.goom import to_goom, from_goom

# Functions for computing log-sums and log-means of exponentials:


def log_add_exp(log_x1: jax.Array, log_x2: jax.Array) -> jax.Array:
    """Elementwise log(exp(log_x1) + exp(log_x2))."""
    c = jnp.maximum(jnp.max(log_x1.real), jnp.max(log_x2.real))
    x = from_goom(log_x1 - c) + from_goom(log_x2 - c)
    return to_goom(x) + c


def log_sum_exp(log_x: jax.Array, axis: int) -> jax.Array:
    """Log-sum-exp over an axis of log_x."""
    c = jnp.max(log_x.real, axis=axis, keepdims=True)
    x = from_goom(log_x - c).sum(axis=axis)
    return to_goom(x) + jnp.squeeze(c, axis=axis)


def log_mean_exp(log_x: jax.Array, axis: int) -> jax.Array:
    """Log-mean-exp over an axis of log_x."""
    log_n = jnp.log(jnp.array(log_x.shape[axis], dtype=log_x.real.dtype))
    return log_sum_exp(log_x, axis=axis) - log_n


# Functions for computing cumulative log-sums and log-means of exponentials:


def log_cum_sum_exp(log_x: jax.Array, axis: int) -> jax.Array:
    """Log-cumulative-sum-exp over an axis of log_x."""
    c = jnp.max(log_x.real, axis=axis, keepdims=True)
    x = from_goom(log_x - c).cumsum(axis=axis)
    return to_goom(x) + c


def log_cum_mean_exp(log_x: jax.Array, axis: int) -> jax.Array:
    """Log-cumulative-mean-exp over an axis of log_x."""
    log_x = jnp.moveaxis(log_x, axis, -1)
    log_n = jnp.log(jnp.arange(1, log_x.shape[-1] + 1, dtype=log_x.real.dtype))
    log_x = log_cum_sum_exp(log_x, axis=-1) - log_n
    return jnp.moveaxis(log_x, -1, axis)


# Convenience functions:


def log_negate_exp(log_x: jax.Array) -> jax.Array:
    """Elementwise negate log_x, a generalized order of magnitude."""
    return log_x + (1j * jnp.pi)


def scale(log_x: jax.Array, axis: int, max_real: float = 2.0) -> jax.Array:
    """Scale log_x so max real component over an axis is max_real."""
    c = jnp.max(log_x.real, axis=axis, keepdims=True)
    return log_x - c + max_real


def scaled_exp(log_x: jax.Array, axis: int, max_real: float = 2.0) -> jax.Array:
    """Scale log_x so max real component over an axis is max_real, and exponentiate."""
    return from_goom(scale(log_x, axis=axis, max_real=max_real))


def log_triu_exp(log_x: jax.Array, diagonal_offset: int = 0) -> jax.Array:
    """Mask log_x in last two dims so its exp() is an upper-triangular matrix."""
    keep_idx = jnp.triu(jnp.ones(log_x.shape[-2:], dtype=bool), k=diagonal_offset)
    real = jnp.where(keep_idx, log_x.real, -jnp.inf)
    return real + 1j * log_x.imag


def log_tril_exp(log_x: jax.Array, diagonal_offset: int = 0) -> jax.Array:
    """Mask log_x in last two dims so its exp() is a lower-triangular matrix."""
    keep_idx = jnp.tril(jnp.ones(log_x.shape[-2:], dtype=bool), k=diagonal_offset)
    real = jnp.where(keep_idx, log_x.real, -jnp.inf)
    return real + 1j * log_x.imag


def log_rmsnorm_exp(log_x: jax.Array, axis: int) -> jax.Array:
    """Scale log_x over an axis so exp() has unit root mean squared elements."""
    log_d = math.log(float(log_x.shape[axis]))
    sum_exp = log_sum_exp(2 * log_x, axis=axis)
    expanded_sum_exp = jnp.expand_dims(sum_exp, axis=axis)
    return log_x - 0.5 * (expanded_sum_exp - log_d)


def log_unitsquash_exp(log_x: jax.Array) -> jax.Array:
    """Elementwise squash log_x so exp() is between -1 and 1."""
    squashed_real = jax.nn.log_sigmoid(log_x.real)
    return squashed_real + 1j * log_x.imag
