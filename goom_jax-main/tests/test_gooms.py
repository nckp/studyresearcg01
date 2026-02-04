import jax
import jax.numpy as jnp

from goom.goom import to_goom, from_goom, generate_random_gooms


def test_generate_random_gooms_no_nans():
    """Test that generate_random_gooms doesn't produce NaNs in real or imag parts."""
    key = jax.random.PRNGKey(0)
    shape = (20, 10)
    gooms = generate_random_gooms(key, shape)
    assert not jnp.isnan(gooms.real).any(), "NaNs found in real part of gooms"
    assert not jnp.isnan(gooms.imag).any(), "NaNs found in imaginary part of gooms"


def test_generate_random_gooms_no_overflows():
    """Test that generate_random_gooms doesn't produce NaNs in real or imag parts."""
    key = jax.random.PRNGKey(0)
    shape = (2048, 2048)
    gooms, floats = generate_random_gooms(key, shape, debug=True, zero_at_zero=True)

    unconverted_floats_range = jnp.max(floats) - jnp.min(floats)
    assert unconverted_floats_range > 0, "Floats should have a range"
    assert (
        unconverted_floats_range > jnp.finfo(jnp.float16).max
    ), "This many positive and negative Floats should have a range greater than the max float value"

    should_have_neginfs = floats == 0

    assert should_have_neginfs[0].all(), "0 should map to realpart -inf in gooms"

    # 0 maps to realpart -inf in gooms, so we need to exclude zeros from the negative inf check
    neginfs = (jnp.isneginf(gooms.real) & ~should_have_neginfs) | jnp.isneginf(
        gooms.imag
    )
    neginfs_found = neginfs.any()
    if neginfs_found:
        print(
            f"Negative inf values {gooms[neginfs]} in gooms generated from floats {floats[neginfs]} at indices: {jnp.argwhere(neginfs)}"
        )
    assert not neginfs_found, "Negative infs found in gooms"

    infs = jnp.isposinf(gooms.real) | jnp.isposinf(gooms.imag)
    infs_found = infs.any()
    if infs_found:
        print(
            f"Positive inf values {gooms[infs]} in gooms generated from floats {floats[infs]} at indices: {jnp.argwhere(infs)}"
        )
    assert not infs_found, "Positive Infs found in gooms"


def test_goom_inversion():
    """Test that from_goom(to_goom(x)) is close to x."""
    key = jax.random.PRNGKey(123)
    shape = (100,)

    # Generate random floats
    minval = jnp.finfo(jnp.float32).min
    maxval = jnp.finfo(jnp.float32).max
    original_floats = jax.random.uniform(
        key, shape, dtype=jnp.float32, minval=minval, maxval=maxval
    )

    # Apply to_goom and then from_goom
    gooms = to_goom(original_floats)
    reconstructed_floats = from_goom(gooms)

    # Check if the reconstructed floats are close to the original ones
    assert jnp.allclose(original_floats, reconstructed_floats, rtol=1e-5, atol=1e-5)
