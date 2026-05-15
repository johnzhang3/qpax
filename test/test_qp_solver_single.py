import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qpax import solve_qp

from .misc_test_utils import check_kkt_conditions, generate_random_qp


@pytest.mark.parametrize("backend", ["e", "i"])
def test_qp_solver(backend):
    np.random.seed(1)

    nx = 15
    ns = 10
    ny = 3

    jit_solve_qp = jax.jit(solve_qp, static_argnames=("backend",))
    solver_tol = 1e-2
    for first_test_iter in range(100):
        Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(
            nx, ns, ny
        )
        x, s, z, y, converged, iters = jit_solve_qp(
            Q, q, A, b, G, h, backend=backend, solver_tol=solver_tol
        )
        print(
            "test iter: ", first_test_iter, "converged: ", converged, "iters: ", iters
        )
        print("x - xreal: ", jnp.linalg.norm(x - x_true))

        del s_true, z_true, y_true

        assert converged == 1
        assert iters <= 10

        check_kkt_conditions(Q, q, A, b, G, h, x, s, z, y, solver_tol=solver_tol)

    ny = 0

    for second_test_iter in range(100):
        Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(
            nx, ns, ny
        )
        x, s, z, y, converged, iters = jit_solve_qp(
            Q, q, A, b, G, h, backend=backend, solver_tol=solver_tol
        )

        print(
            "test iter: ", second_test_iter, "converged: ", converged, "iters: ", iters
        )
        print("x - xreal: ", jnp.linalg.norm(x - x_true))

        del s_true, z_true, y_true

        assert converged == 1
        assert iters <= 10

        check_kkt_conditions(Q, q, A, b, G, h, x, s, z, y, solver_tol=solver_tol)
