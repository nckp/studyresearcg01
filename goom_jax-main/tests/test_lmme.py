"""Tests for the matrix exponential sum implementation."""

import numpy as np
import jax
import jax.numpy as jnp

from goom.lmme import lmme_impls
from goom.goom import from_goom, to_goom, generate_random_gooms


def test_lmme_equivalence():
    """Test that all lmme implementations give similar results."""
    key = jax.random.PRNGKey(42)
    n, d, m = 8, 7, 6
    key, subkey1 = jax.random.split(key)
    log_x1 = generate_random_gooms(subkey1, (n, d))
    key, subkey2 = jax.random.split(key)
    log_x2 = generate_random_gooms(subkey2, (d, m))

    # Run all implementations
    results = {}
    for impl in lmme_impls:
        result = impl(log_x1, log_x2)
        assert not jnp.isnan(
            result.real
        ).any(), f"NaNs found in real part of {impl.__name__}"
        assert not jnp.isnan(
            result.imag
        ).any(), f"NaNs found in imaginary part of {impl.__name__}"
        results[impl.__name__] = result

    # Compare all pairs of implementations
    impl_names = list(results.keys())
    for i in range(len(impl_names)):
        for j in range(i + 1, len(impl_names)):
            name1, name2 = impl_names[i], impl_names[j]
            result1 = results[name1]
            result2 = results[name2]

            # it only matters that things match back in R space, not goom space
            result1 = from_goom(result1)
            result2 = from_goom(result2)

            # Check if arrays are (kinda) close
            # since we are comparing ALL implementations, some of which might not
            # be shooting for maximum precision, we only care about relative error
            # and with pretty fat tolerances at that
            is_close = jnp.isclose(result1, result2, rtol=1e-2, atol=0)

            if not jnp.all(is_close):
                # Find indices where arrays don't match
                mismatch_mask = ~is_close
                mismatch_indices = np.argwhere(np.asarray(mismatch_mask))
                num_mismatches_total = len(mismatch_indices)

                # Get first 5 non-matching values
                num_mismatches = min(5, num_mismatches_total)
                error_msg = f"Results from {name1} and {name2} do not match.\n"
                error_msg += f"Found {num_mismatches_total} non-matching values. First {num_mismatches}:\n"

                for idx in range(num_mismatches):
                    idx_tuple = tuple(mismatch_indices[idx])
                    val1 = result1[idx_tuple]
                    val2 = result2[idx_tuple]
                    diff = val1 - val2
                    error_msg += f"  Index {idx_tuple}: {name1}={val1}, {name2}={val2}, diff={diff}\n"

                assert False, error_msg
