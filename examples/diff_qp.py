"""Differentiate through a QP solve with qpax.

This example solves a non-negative least-squares problem

    minimize_x ||F x - g||^2
    subject to x >= 0

after converting it to standard QP form, then differentiates the loss

    L(x) = ||x - 1||^2

with respect to the QP data.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

import qpax

N_VARS = 5
N_ROWS = 10


def loss(Q, q, A, b, G, h):
    x = qpax.solve_qp_primal(Q, q, A, b, G, h)
    x_target = jnp.ones_like(x)
    return jnp.sum((x - x_target) ** 2)


def nnls_to_qp(F, g):
    n_vars = F.shape[1]
    Q = F.T @ F
    q = -F.T @ g
    A = jnp.zeros((0, n_vars))
    b = jnp.zeros(0)
    G = -jnp.eye(n_vars)
    h = jnp.zeros(n_vars)
    return Q, q, A, b, G, h


# Create one random NNLS problem.
F = jnp.array(np.random.randn(N_ROWS, N_VARS))
g = jnp.array(np.random.randn(N_ROWS))

# Convert it to QP form.
Q, q, A, b, G, h = nnls_to_qp(F, g)

# Evaluate the loss and its gradients.
loss_and_grad = jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5)))
loss_value, derivs = loss_and_grad(Q, q, A, b, G, h)
dl_dQ, dl_dq, dl_dA, dl_db, dl_dG, dl_dh = derivs

print("loss:", float(loss_value))
print("dl_dQ.shape:", dl_dQ.shape)
print("dl_dq.shape:", dl_dq.shape)
print("dl_dG.shape:", dl_dG.shape)
print("dl_dh.shape:", dl_dh.shape)
