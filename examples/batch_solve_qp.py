"""Solve a batch of non-negative least-squares problems with qpax.

Each problem is

    minimize_x ||F x - g||^2
    subject to x >= 0
"""

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

import qpax

N_VARS = 5
N_ROWS = 10
N_QPS = 10_000


@jit
def nnls_to_qp(F, g):
    """Convert one NNLS problem to standard QP form."""
    n_vars = F.shape[1]
    Q = F.T @ F
    q = -F.T @ g
    A = jnp.zeros((0, n_vars))
    b = jnp.zeros(0)
    G = -jnp.eye(n_vars)
    h = jnp.zeros(n_vars)
    return Q, q, A, b, G, h


def solve_one_qp(Q, q, A, b, G, h):
    return qpax.solve_qp(Q, q, A, b, G, h)


# Create a batch of random NNLS problems.
Fs = jnp.array(np.random.randn(N_QPS, N_ROWS, N_VARS))
gs = jnp.array(np.random.randn(N_QPS, N_ROWS))

# Convert the whole batch to QP form.
batch_nnls_to_qp = vmap(nnls_to_qp, in_axes=(0, 0))
Qs, qs, As, bs, Gs, hs = batch_nnls_to_qp(Fs, gs)

# Solve all QPs in parallel and keep the convergence diagnostics.
batch_solve_qp = jit(vmap(solve_one_qp, in_axes=(0, 0, 0, 0, 0, 0)))
xs, _, _, _, converged, iters = batch_solve_qp(Qs, qs, As, bs, Gs, hs)

converged = np.asarray(converged)
iters = np.asarray(iters)

print("xs.shape:", xs.shape)
print(f"converged: {int(converged.sum())}/{N_QPS}")
print(f"median iterations: {np.median(iters):.1f}")
print(f"max iterations: {iters.max()}")
