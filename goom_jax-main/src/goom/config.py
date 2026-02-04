from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class Config:

    keep_logs_finite: bool = True
    "If True, log() always returns finite values. Otherwise, it does not."

    cast_all_logs_to_complex: bool = True
    "If True, to_goom() always returns complex tensors. Otherwise, it does not."

    # float_dtype: jnp.dtype = jnp.float32
    "Float dtype of real logarithms and components of complex logarithms."


config = Config()
