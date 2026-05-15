# Installation

`qpax` runs on CPU out of the box and on NVIDIA GPUs through JAX's CUDA
wheels. Pick the variant that matches your hardware, or install from source
if you plan to read or modify the code.

## Quick install

Install the latest release from PyPI with `pip`:

=== "CPU"

    ```bash
    pip install qpax
    ```

=== "NVIDIA GPU (CUDA 12)"

    ```bash
    pip install "qpax[cuda12]"
    ```

=== "NVIDIA GPU (CUDA 13)"

    ```bash
    pip install "qpax[cuda13]"
    ```

The GPU variants pull in the matching `jax[cuda12]` / `jax[cuda13]` extras.
You only need one — match it to the CUDA toolkit on your machine.

## Installing from source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/qpax-solver/qpax
cd qpax
```

Then pick the variant you want:

=== "Minimal"

    ```bash
    pip install -e .
    ```

=== "Development"

    Includes test, linting, and documentation dependencies.

    ```bash
    pip install -e ".[dev]"
    ```

=== "NVIDIA GPU (CUDA 12)"

    ```bash
    pip install -e ".[cuda12]"
    ```

=== "NVIDIA GPU (CUDA 13)"

    ```bash
    pip install -e ".[cuda13]"
    ```

## Verify installation

Run the test suite from the repository root:

```bash
python -m pytest test
```

A clean run prints a summary line ending in `passed`. If you only want a
quick sanity check, this single import is enough:

```python
import qpax
import jax.numpy as jnp

Q = jnp.eye(2)
q = jnp.array([1.0, -1.0])
A = jnp.zeros((0, 2)); b = jnp.zeros(0)
G = -jnp.eye(2);       h = jnp.zeros(2)

x, *_ = qpax.solve_qp(Q, q, A, b, G, h)
print(x)
```
