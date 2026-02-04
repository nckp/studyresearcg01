import jax.numpy as jnp
from jax import jax, vmap
import timeit

from goom.goom import from_goom, goom_exp, goom_log, to_goom


def log_matmul_exp(log_x1: jax.Array, log_x2: jax.Array, min_c: int = 0) -> jax.Array:
    """
    Broadcastable log(exp(log_x1) @ exp(log_x2)), implemented by calling
    JAX's existing implementation of matmul over scaled float tensors.
    Inputs:
        log_x1: log-tensor of shape [..., d1, d2].
        log_x2: log-tensor of shape [..., d2, d3].
        min_c: (optional) float, minimum log-scaling constant. Default: 0.
    Outputs:
        log_y: log-tensor of shape [..., d1, d3].
    """
    c1 = jnp.maximum(jnp.max(log_x1.real, axis=-1, keepdims=True), min_c)
    c2 = jnp.maximum(jnp.max(log_x2.real, axis=-2, keepdims=True), min_c)

    x1 = from_goom(log_x1 - c1)
    x2 = from_goom(log_x2 - c2)

    scaled_y = jnp.matmul(x1, x2)

    log_y = to_goom(scaled_y) + c1 + c2
    return log_y


def alternate_log_matmul_exp(log_x1: jax.Array, log_x2: jax.Array) -> jax.Array:
    """
    Broadcastable log(exp(log_x1) @ exp(log_x2)), implemented by composition
    of vmapped log-sum-exp-of-sum operations. Much slower, but more precise.
    Inputs:
        log_x1: log-tensor of shape [..., d1, d2].
        log_x2: log-tensor of shape [..., d2, d3].
    Outputs:
        log_y: log-tensor of shape [..., d1, d3].
    """
    # Get log-scaling constants:
    c1 = jnp.max(log_x1.real, axis=-1, keepdims=True)
    c2 = jnp.max(log_x2.real, axis=-2, keepdims=True)

    # Get log-scaled operands:
    log_s1 = log_x1 - c1
    log_s2 = log_x2 - c2

    # Broadcast preceding dims and flatten them into a single dim:
    d1, d2, d3 = (*log_s1.shape[-2:], log_s2.shape[-1])
    broadcast_szs = jnp.broadcast_shapes(log_s1.shape[:-2], log_s2.shape[:-2])

    log_s1_flat = jnp.broadcast_to(log_s1, broadcast_szs + (d1, d2)).reshape(-1, d1, d2)
    log_s2_flat = jnp.broadcast_to(log_s2, broadcast_szs + (d2, d3)).reshape(-1, d2, d3)

    # Define vmapped sum-exp-of-outer-sum operations:
    _vve = lambda row_vec, col_vec: from_goom(row_vec + col_vec).sum()
    _mve = jax.vmap(_vve, (0, None))
    _mme = jax.vmap(_mve, (None, 1), 1)
    _multi_mme = jax.vmap(_mme)

    # Compute, reshape, and return result:
    scaled_y = _multi_mme(log_s1_flat, log_s2_flat)
    log_y = to_goom(scaled_y) + c1 + c2
    return log_y.reshape(*broadcast_szs, d1, d3)


def vmap_sum_log_matmul_exp(A, B):
    """
    vmap-based implementation for moderate sizes.
    """

    def compute_row(a_row, B):
        terms = goom_exp(a_row[:, jnp.newaxis] + B)
        return jnp.sum(terms, axis=0)

    sums = vmap(compute_row, in_axes=(0, None))(A, B)
    return goom_log(sums)


lmme_impls = [log_matmul_exp, alternate_log_matmul_exp, vmap_sum_log_matmul_exp]
N = 64
base_shape = (N, N, N)
n, d, m = base_shape
powers = [2**i for i in range(3)]
shapes = [(n * power, d * power, m * power) for power in powers]


def compare_lmme_implementations(key, shape=(3, 2, 4)):
    """
    Compares the outputs of different log_matmul_exp implementations.
    """
    n, d, m = shape
    key, subkey1, subkey2 = jax.random.split(key, 3)

    from goom.goom import generate_random_gooms

    log_x1 = generate_random_gooms(subkey1, (n, d))
    log_x2 = generate_random_gooms(subkey2, (d, m))

    results = {impl.__name__: impl(log_x1, log_x2) for impl in lmme_impls}

    # Formatting for printing
    for name, result in results.items():
        print(f"--- {name} ---")
        for row in result:
            row_str = "  ".join([f"{x.real:7.2f}{x.imag/jnp.pi:+.2f}πj" for x in row])
            print(f"[ {row_str} ]")
        print()


def benchmark_lmme_implementations(key, repeats=5, number=10, shapes=shapes):
    """
    Measures and prints the running times for each lmme implementation across various shapes.
    """
    from goom.goom import generate_random_gooms

    timings = {shape: {} for shape in shapes}

    for shape in shapes:
        print(f"\nBenchmarking shape: {shape}...")
        n, d, m = shape

        data_key, key = jax.random.split(key)
        log_x1 = generate_random_gooms(data_key, (n, d))
        log_x2 = generate_random_gooms(data_key, (d, m))

        for impl in lmme_impls:
            try:
                jitted_impl = jax.jit(impl)
                jitted_impl(log_x1, log_x2).block_until_ready()

                stmt = "jitted_impl(log_x1, log_x2).block_until_ready()"
                timer = timeit.Timer(
                    stmt=stmt,
                    globals={
                        "jitted_impl": jitted_impl,
                        "log_x1": log_x1,
                        "log_x2": log_x2,
                    },
                )

                time_list = timer.repeat(repeat=repeats, number=number)
                best_time = min(time_list) / number

                timings[shape][impl.__name__] = best_time
                print(f"  {impl.__name__:<30}: {best_time:.6f}s")

            except Exception as e:
                print(f"  {impl.__name__:<30}: FAILED ({e})")
                timings[shape][impl.__name__] = float("inf")


if __name__ == "__main__":
    main_key = jax.random.PRNGKey(0)
    # compare_lmme_implementations(main_key)
    benchmark_lmme_implementations(main_key)
