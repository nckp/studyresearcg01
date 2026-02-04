import jax
import jax.numpy as jnp
from jax import lax

import sys
import goom as goom

print("goom module:", goom)
print("Python executable:", sys.executable)
print("Python version:", sys.version)

import goom.lmme as lmme
import goom.operations as oprs

import time
import numpy as np


def randn_like(x, key, dtype=None):
    return jax.random.normal(key, shape=x.shape, dtype=(dtype or x.dtype))


def normalize(x, axis=-1, eps=1e-12):
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def rand_like_normalized(jac_vals, axis, key):
    return normalize(
        randn_like(jac_vals[..., 0, :1, :], key),
        axis=axis,
    )


def jax_estimate_lle_parallel(jac_vals, key, dt=1.0):
    
    """
    Estimate the largest Lyapunov exponent from Jacobians, in parallel
    
    jacobians: array of shape (T, D, D)
               J[t] is the Jacobian at time step t
    key:       jax.random.PRNGKey
    dt:        time step between Jacobians (default 1.0)
    
    """
    # get log_jac_vals
    T = jac_vals.shape[-3]
    log_jac_vals = goom.goom.to_goom(jac_vals)

    # initialize random unit vector u[0]
    key, u0_key = jax.random.split(key)
    u0 = rand_like_normalized(jac_vals, axis=-1, key=u0_key)

    # multiply Jacobians from last to first: M[T] = J[T] @ ... @ J[0] in goom space and grab last cumulative product
    log_jac_product = lax.associative_scan(
        lmme.log_matmul_exp, jnp.flip(log_jac_vals, axis=0), axis=0
    )[-1]

    # M[T] @ u[0] in goom space
    log_end_state = lmme.log_matmul_exp(u0, log_jac_product)

    # get final LLE estimate
    lambda_max = oprs.log_sum_exp(log_end_state * 2, axis=-1).real / (2 * T * dt)

    return lambda_max[0]


def jax_estimate_lle_sequential(jacobians, key, dt=1.0, eps=1e-12):
    """
    jacobians: array of shape (T, D, D)
               J[t] is the Jacobian at time step t
    key:       jax.random.PRNGKey
    dt:        time step between Jacobians (default 1.0)
    """
    T, D, _ = jacobians.shape

    # random initial unit vector in R^D
    v0 = jax.random.normal(key, shape=(D,))
    v0 = normalize(v0, axis=0, eps=eps)

    def step(v, J):
        # propagate tangent vector
        w = J @ v
        norm = jnp.linalg.norm(w) + eps
        v_next = w / norm
        log_norm = jnp.log(norm)
        return v_next, log_norm

    # run sequentially over time with lax.scan (JAX-friendly loop)
    vT, log_norms = lax.scan(step, v0, jacobians)

    # average growth rate -> largest Lyapunov exponent
    # divide by dt if your Jacobians correspond to time step dt
    lambda_max = jnp.mean(log_norms) / dt
    return lambda_max



def benchmark_lorenz_lle(
    key,
    T=400_000,
    dt=1e-3,
    n_runs=5,
):
    """
    Lorenz + GOOM LLE benchmarking function.

    IMPORTANT: This benchmark measures ONLY the cost of computing the
    Lyapunov exponent from a precomputed sequence of Jacobians.

    - Lorenz system + Euler step used only to precompute Jacobians.
    - simulate_lorenz_with_jacobians() is run ONCE and NOT benchmarked.
    - LLE functions are assumed to take Jacobians as their input:
        jax_estimate_lle_parallel(jacobians, key, dt)
        jax_estimate_lle_sequential(jacobians, key, dt)
    """

    # ----------------------------
    # GOOM global config
    # ----------------------------
    cfg = goom.config
    cfg.keep_logs_finite = True
    cfg.cast_all_logs_to_complex = True
    cfg.float_dtype = jnp.float32

    # ----------------------------
    # Lorenz system + Euler step
    # ----------------------------
    def lorenz(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        x_, y_, z_ = x
        dx = sigma * (y_ - x_)
        dy = x_ * (rho - z_) - y_
        dz = x_ * y_ - beta * z_
        return jnp.array([dx, dy, dz])

    def euler_step(x, dt=dt):
        return x + dt * lorenz(x)

    # ----------------------------
    # Simulator that PRECOMPUTES Jacobians (not benchmarked)
    # ----------------------------
    def simulate_lorenz_with_jacobians(x0, T):
        """
        Run the Lorenz system and return (states, Jacobians) for each step.

        This is used ONLY to generate the Jacobians that will be passed
        into the LLE estimators. Its runtime is not included in the
        parallel vs sequential LLE benchmark.
        """
        def step_with_jac(x, _):
            def step_fn(x_):
                return euler_step(x_, dt=dt)

            x_next = step_fn(x)
            J = jax.jacrev(step_fn)(x)
            return x_next, (x_next, J)

        _, (xs, Js) = lax.scan(step_with_jac, x0, None, length=T)
        states = jnp.vstack([x0[None, :], xs])
        return states, Js

    simulate_lorenz_with_jacobians_jit = jax.jit(
        simulate_lorenz_with_jacobians,
        static_argnames=("T",),
    )

    # ----------------------------
    # Benchmark helper
    # ----------------------------
    def benchmark(f, *args, n_runs=5, key_arg_index=None, base_key=None, **kwargs):
        """
        Benchmark `f` on given args.

        - Uses median of `n_runs` wall-clock times.
        - If `key_arg_index` and `base_key` are provided, it will
          use `jax.random.fold_in(base_key, i)` for run i, and
          place that key into args[key_arg_index] each run.
        """
        times = []
        last_out = None

        for i in range(n_runs):
            run_args = list(args)
            if key_arg_index is not None and base_key is not None:
                run_args[key_arg_index] = jax.random.fold_in(base_key, i)

            start = time.perf_counter()
            out = f(*run_args, **kwargs)
            out = jax.block_until_ready(out)
            end = time.perf_counter()

            times.append(end - start)
            last_out = out

        median_time = float(np.median(times))
        return median_time, last_out

    # ----------------------------
    # Precompute Jacobians (NOT benchmarked)
    # ----------------------------
    x0 = jnp.array([1.0, 0.0, 1.0])

    print(f"Simulating Lorenz system and precomputing Jacobians for T={T}...")
    states, jac_vals = simulate_lorenz_with_jacobians_jit(x0, T)
    # We keep `states` only if you need them for sanity checks or later use.
    # The LLE benchmark below uses ONLY `jac_vals`.

    # ----------------------------
    # Benchmark LLE from Jacobians only
    # ----------------------------
    # Key split for parallel vs sequential LLE runs
    key_par_base, key_seq_base = jax.random.split(key)

    # JIT the LLE functions, which now assume "jacobians-first" signature
    lle_par_jit = jax.jit(jax_estimate_lle_parallel)
    lle_seq_jit = jax.jit(jax_estimate_lle_sequential)

    # Warm-up to trigger compilation (not timed)
    _ = lle_par_jit(jac_vals, key_par_base, dt=dt).block_until_ready()
    _ = lle_seq_jit(jac_vals, key_seq_base, dt=dt).block_until_ready()

    print("\nBenchmarking parallel LLE (Jacobian-only)...")
    t_par, est_par = benchmark(
        lle_par_jit,
        jac_vals,
        key_par_base,          # placeholder; overwritten inside benchmark
        dt=dt,
        n_runs=n_runs,
        key_arg_index=1,       # index of key in (jac_vals, key, dt)
        base_key=key_par_base,
    )

    print("Benchmarking sequential LLE (Jacobian-only)...")
    t_seq, est_seq = benchmark(
        lle_seq_jit,
        jac_vals,
        key_seq_base,
        dt=dt,
        n_runs=n_runs,
        key_arg_index=1,
        base_key=key_seq_base,
    )

    speedup = t_seq / t_par

    print("\n=== Lorenz + GOOM LLE Benchmark Results (Jacobian-only) ===")
    print(f"Parallel LLE:   {float(est_par):.4f} | median_time={t_par:.4f}s")
    print(f"Sequential LLE: {float(est_seq):.4f} | median_time={t_seq:.4f}s")
    print(
        f"Speedup (sequential_time / parallel_time): "
        f"{speedup:.2f}x"
    )

    return {
        "parallel": {"time": t_par, "lle": est_par},
        "sequential": {"time": t_seq, "lle": est_seq},
        "speedup": speedup,
    }


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    print("Running LLE benchmark (Jacobian-only). Might take a minute for T=400_000.")
    results = benchmark_lorenz_lle(key)