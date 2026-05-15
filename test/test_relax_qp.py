import jax
import numpy as np
import pytest

from qpax import relax_qp, solve_qp

from .misc_test_utils import (
    check_relaxed_implicit_conditions,
    generate_random_qp,
)


@pytest.mark.parametrize("backend", ["e", "i"])
def test_relax_qp(backend):
    np.random.seed(5)

    jit_solve_qp = jax.jit(solve_qp, static_argnames=("backend",))
    jit_relax_qp = jax.jit(relax_qp, static_argnames=("backend",))

    nx = 15
    ns = 10
    ny = 3
    solver_tol = 1e-5
    target_kappa = 1e-4

    for test_iter in range(100):
        Q, q, A, b, G, h, x_true, s_true, z_true, y_true = generate_random_qp(
            nx, ns, ny
        )
        x, s, z, y, converged, iters = jit_solve_qp(
            Q, q, A, b, G, h, backend=backend, solver_tol=solver_tol
        )

        del x_true, s_true, z_true, y_true, iters

        assert converged == 1

        xr, sr, zr, yr, relaxed_converged, relaxed_iters = jit_relax_qp(
            Q,
            q,
            A,
            b,
            G,
            h,
            x,
            s,
            z,
            y,
            backend=backend,
            solver_tol=solver_tol,
            target_kappa=target_kappa,
        )

        print(
            "test iter: ",
            test_iter,
            "relaxed converged: ",
            relaxed_converged,
            "iters: ",
            relaxed_iters,
        )

        assert relaxed_converged == 1

        check_relaxed_implicit_conditions(
            Q,
            q,
            A,
            b,
            G,
            h,
            xr,
            sr,
            zr,
            yr,
            target_kappa,
            solver_tol=solver_tol,
        )
