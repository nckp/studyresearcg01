import jax
import jax.numpy as jnp
import math
from typing import Tuple

from goom.config import config

# Custom gradient functions


@jax.custom_vjp
@jax.jit
def goom_abs(x: jax.Array) -> jax.Array:
    """Absolute value with a custom derivative that equals 1 at zero."""
    return jnp.abs(x)


def _goom_abs_fwd(x: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Forward pass that keeps the original input as residuals."""
    y = jnp.abs(x)
    residuals = x
    return y, residuals


def _goom_abs_bwd(residuals: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    """Backward pass that defines grad(|x|) = sign(x), but 1 when x == 0."""
    x = residuals
    grad_x = g * jnp.where(x == 0, jnp.ones_like(x), jnp.sign(x))
    return (grad_x,)


goom_abs.defvjp(_goom_abs_fwd, _goom_abs_bwd)


@jax.custom_vjp
def goom_exp(x: jax.Array) -> jax.Array:
    """Exponentiate ``x`` while providing a custom gradient."""
    return jnp.exp(x)


def _goom_exp_fwd(x: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Forward pass returns the primal value and residuals."""
    y = jnp.exp(x)
    residuals = x
    return y, residuals


def _goom_exp_bwd(residuals: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    """Backward pass that perturbs the residuals with a signed epsilon."""
    x = residuals
    eps = jnp.finfo(g.real.dtype).eps
    signed_eps = jnp.where(x.real < 0, -eps, eps)
    grad_x = g * (x + signed_eps)
    return (grad_x,)


goom_exp.defvjp(_goom_exp_fwd, _goom_exp_bwd)


@jax.custom_vjp
def goom_log(x: jax.Array) -> jax.Array:
    """Logarithm of ``x`` while providing a custom gradient."""
    return jnp.log(x)


def _goom_log_fwd(x: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Forward pass returns the primal value and residuals."""
    log_inp = jnp.log(x)
    snn = jnp.finfo(x.real.dtype).smallest_normal
    finite_floor = math.log(snn) * 2  # exps to zero in float_dtype
    keep_finite_idx = (log_inp < finite_floor) & config.keep_logs_finite
    out = jnp.where(keep_finite_idx, finite_floor, log_inp)
    residuals = x
    return out, residuals


def _goom_log_bwd(residuals: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    """Backward pass that perturbs the residuals with a signed epsilon."""
    x = residuals
    eps = jnp.finfo(g.real.dtype).eps
    grad_x = g / (x + eps)
    return (grad_x,)


goom_log.defvjp(_goom_log_fwd, _goom_log_bwd)


def to_goom(x: jax.Array) -> jax.Array:
    abs_x = goom_abs(x)
    log_abs_x = goom_log(abs_x)
    real = jnp.astype(log_abs_x, jnp.float32)
    x_is_neg = x < 0
    # todo: allow this to return real tensors if config.cast_all_logs_to_complex is False and x is
    # non-negative
    return jnp.complex64(real + 1j * x_is_neg * jnp.pi)


def from_goom(x: jax.Array) -> jax.Array:
    return goom_exp(x).real


def generate_random_gooms(
    key: jax.Array,
    shape: tuple[int, ...],
    debug: bool = False,
    zero_at_zero: bool = False,
) -> jax.Array | Tuple[jax.Array, jax.Array]:
    minval = jnp.finfo(jnp.float16).min
    maxval = jnp.finfo(jnp.float16).max
    floats = jax.random.uniform(
        key, shape, dtype=jnp.float32, minval=minval, maxval=maxval
    )
    if zero_at_zero:
        floats = floats.at[0].set(0.0)
    gooms = to_goom(floats)
    if debug:
        return gooms, floats
    return gooms
