"""Differentiate through a batch of QP solves with qpax.

Each problem is a non-negative least-squares problem

    minimize_x ||F x - g||^2
    subject to x >= 0

after converting it to standard QP form. For each QP in the batch, this
example differentiates the loss

    L(x) = ||x - 1||^2

with respect to the QP data.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

import qpax

N_VARS = 5
N_ROWS = 10
N_QPS = 256


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


# Create a batch of random NNLS problems.
Fs = jnp.array(np.random.randn(N_QPS, N_ROWS, N_VARS))
gs = jnp.array(np.random.randn(N_QPS, N_ROWS))

# Convert them to QP form.
batch_nnls_to_qp = vmap(nnls_to_qp, in_axes=(0, 0))
Qs, qs, As, bs, Gs, hs = batch_nnls_to_qp(Fs, gs)

# Evaluate the loss and its gradients for all QPs in parallel.
loss_and_grad = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5))
batch_loss_and_grad = jit(vmap(loss_and_grad, in_axes=(0, 0, 0, 0, 0, 0)))
losses, derivs = batch_loss_and_grad(Qs, qs, As, bs, Gs, hs)
dl_dQ, dl_dq, dl_dA, dl_db, dl_dG, dl_dh = derivs

print("losses.shape:", losses.shape)
print("mean loss:", float(jnp.mean(losses)))
print("dl_dQ.shape:", dl_dQ.shape)
print("dl_dq.shape:", dl_dq.shape)
print("dl_dG.shape:", dl_dG.shape)
print("dl_dh.shape:", dl_dh.shape)
