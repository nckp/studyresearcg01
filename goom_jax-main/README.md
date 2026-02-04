JAX version of [generalized_orders_of_magnitude](https://github.com/glassroom/generalized_orders_of_magnitude).

Note: This code has not been profiled or optimized yet.

## Setup

``` bash
uv sync --extra dev
```

## Run the tests

``` bash
uv run --extra dev pytest
```

## Run some benchmarks

Benchmark basic log_matmul_exp (LMME) functionality
``` bash
uv run python -m goom.lmme 
```

Benchmark GPU speedup for Largest Lyapunov Exponent (LLE) computation. Make sure you have a GPU.
``` bash
uv run python -m goom.lle
```