"""Solve one random convex QP with qpax.

The problem is

    minimize_x 0.5 x^T Q x + q^T x
    subject to A x = b
               G x <= h
"""

import jax
import jax.numpy as jnp
import numpy as np

import qpax

USE_F64 = False
if USE_F64:
    jax.config.update("jax_enable_x64", True)

dtype = jnp.float64 if USE_F64 else jnp.float32


def generate_random_qp(n, m, p, dtype):
    rng = np.random.default_rng(0)

    M = rng.normal(size=(n, n))
    Q = M.T @ M + 1e-2 * np.eye(n)
    q = rng.normal(size=n)

    A = rng.normal(size=(m, n))
    x_ref = rng.normal(size=n)
    b = A @ x_ref

    G = rng.normal(size=(p, n))
    h = G @ x_ref + rng.uniform(0.5, 1.5, size=p)

    return (jnp.asarray(arr, dtype=dtype) for arr in (Q, q, A, b, G, h))


# Build one random feasible QP.
Q, q, A, b, G, h = generate_random_qp(n=3, m=1, p=2, dtype=dtype)

# Solve it and keep the convergence diagnostics.
x, _, _, _, converged, iters = qpax.solve_qp(Q, q, A, b, G, h, verbose=True)

print("converged:", int(converged))
print("iterations:", int(iters))
print("x:", x)
