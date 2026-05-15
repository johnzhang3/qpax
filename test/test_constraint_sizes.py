"""Test constraint sizes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpax import solve_qp

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("backend", ["e", "i"])
def test_no_constraints(backend):
    M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
    P = np.dot(M.T, M)
    q = np.dot(np.array([3.0, 2.0, 3.0]), M).reshape((3,))
    A = np.zeros((0, 3))
    b = np.zeros((0,))
    G = np.zeros((0, 3))
    h = np.zeros((0,))

    x, s, z, y, converged, pdip_iter = jax.jit(solve_qp, static_argnames=("backend",))(
        P, q, A, b, G, h, backend=backend
    )

    del s, z, y

    assert converged == 1
    assert pdip_iter == 0
    assert jnp.linalg.norm(P @ x + q, ord=jnp.inf) < 1e-10


@pytest.mark.parametrize("backend", ["e", "i"])
def test_eq_only(backend):
    M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
    P = np.dot(M.T, M)
    q = np.dot(np.array([3.0, 2.0, 3.0]), M).reshape((3,))
    A = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0])
    G = np.zeros((0, 3))
    h = np.zeros((0,))

    x, s, z, y, converged, pdip_iter = jax.jit(solve_qp, static_argnames=("backend",))(
        P, q, A, b, G, h, backend=backend
    )

    del s, z, pdip_iter

    A = jnp.atleast_2d(A)
    At = jnp.atleast_2d(A.T)
    kkt_mat = jnp.block([[P, At], [A, jnp.zeros((A.shape[0], A.shape[0]))]])
    kkt_vec = jnp.concatenate([-q, b])
    real_sol = jnp.linalg.solve(kkt_mat, kkt_vec)
    x_sol = real_sol[: len(x)]
    y_sol = real_sol[len(x) :]

    print(jnp.linalg.norm(x - x_sol, ord=jnp.inf))
    print(jnp.linalg.norm(y - y_sol, ord=jnp.inf))

    assert converged == 1
    assert (
        jnp.linalg.norm(
            x - jnp.array([0.28026906, -1.55156951, 2.27130045]), ord=jnp.inf
        )
        < 1e-5
    )


@pytest.mark.parametrize("backend", ["e", "i"])
def test_QPSUT03_problem(backend):
    P = jnp.array(
        [
            [122.0, 59.0, 39.0, 9.0],
            [59.0, 95.0, 48.0, 24.0],
            [39.0, 48.0, 26.0, 19.0],
            [9.0, 24.0, 19.0, 90.0],
        ]
    )
    q = jnp.array([66.0, 93.0, 47.0, 11.0])
    A = jnp.zeros((0, 4))
    b = jnp.zeros(0)
    G = jnp.array(
        [
            [-1.0, -0.0, -0.0, -0.0],
            [-0.0, -1.0, -0.0, -0.0],
            [-0.0, -0.0, -1.0, -0.0],
            [-0.0, -0.0, -0.0, -1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    h = jnp.array([np.inf, 0.4, np.inf, 1.0, np.inf, np.inf, 0.5, 1.0])

    x, s, z, y, converged, pdip_iter = jax.jit(solve_qp, static_argnames=("backend",))(
        P, q, A, b, G, h, backend=backend
    )

    del s, z, y, pdip_iter

    assert converged == 1
    assert (
        jnp.linalg.norm(
            x - jnp.array([0.18143455, 0.00843864, -2.35442995, 0.35443034]),
            ord=jnp.inf,
        )
        < 1e-4
    )


@pytest.mark.parametrize("backend", ["e", "i"])
def test_maros_meszaros(backend):
    P = np.array([[8.0, 2.0], [2.0, 10.0]])
    q = np.array([1.5, -2.0])
    A = np.array([]).reshape(0, 2)
    b = np.array([])
    G = np.array(
        [
            [-1.0, 2.0],
            [-2.0, -1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    h = np.array([6.0, -2.0, 0.0, 0.0, 20.0, np.inf])

    x, s, z, y, converged, pdip_iter = jax.jit(solve_qp, static_argnames=("backend",))(
        P, q, A, b, G, h, backend=backend
    )

    del s, z, y, pdip_iter

    assert converged == 1
    assert jnp.linalg.norm(x - jnp.array([0.7625, 0.475]), ord=jnp.inf) < 1e-4


@pytest.mark.parametrize("backend", ["e", "i"])
def test_QPTEST(backend):
    Q = np.array([[8.0, 2.0], [2.0, 10.0]])
    q = np.array([1.5, -2.0])
    G = np.array([[-1.0, 2.0], [-2.0, -1.0]])
    h = np.array([6.0, -2.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([20.0, np.inf])

    G = jnp.vstack((G, jnp.eye(2), -jnp.eye(2)))
    h = jnp.concatenate((h, ub, -lb))
    A = jnp.zeros((0, 2))
    b = jnp.zeros(0)

    x, s, z, y, converged, iters = jax.jit(solve_qp, static_argnames=("backend",))(
        Q, q, A, b, G, h, backend=backend, solver_tol=1e-6
    )

    del iters

    r1 = Q @ x + q + A.T @ y + G.T @ z
    r2 = s * z
    r3 = G @ x + s - h
    r4 = A @ x - b

    kkt_res = jnp.concatenate((r1, r2, r3, r4))
    kkt_res = jnp.where(jnp.isnan(kkt_res), 0, kkt_res)
    assert jnp.linalg.norm(kkt_res, ord=jnp.inf) < 1e-5
    assert converged
