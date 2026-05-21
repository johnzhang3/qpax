import jax
import jax.numpy as jnp

from qpax.explicit.pdip import (
    LinearSolver,
    factorize_full_kkt,
    factorize_kkt,
    solve_full_kkt_rhs,
    solve_kkt_rhs,
    solve_qp,
)
from qpax.explicit.pdip_relaxed import relax_qp


def optnet_derivatives(dz, dlam, dnu, z, lam, nu):
    dl_dQ = 0.5 * (jnp.outer(dz, z) + jnp.outer(z, dz))
    dl_dA = jnp.outer(dnu, z) + jnp.outer(nu, dz)
    dl_dG = jnp.diag(lam) @ (jnp.outer(dlam, z) + jnp.outer(lam, dz))

    dl_dq = dz
    dl_db = -dnu
    dl_dh = -lam * dlam

    return dl_dQ, dl_dq, dl_dA, dl_db, dl_dG, dl_dh


def diff_qp(
    Q,
    q,
    A,
    b,
    G,
    h,
    z,
    s,
    lam,
    nu,
    dl_dz,
    linear_solver: LinearSolver = LinearSolver.CHOLESKY,
    full_kkt: bool = False,
):
    ns = len(h)
    nnu = len(b)
    cotangent_dtype = dl_dz.dtype

    if full_kkt:
        factor = factorize_full_kkt(Q, G, A, s, lam, linear_solver)
        dz, _, dlam_tilde, dnu = solve_full_kkt_rhs(
            G,
            A,
            factor,
            -dl_dz,
            jnp.zeros(ns, dtype=cotangent_dtype),
            jnp.zeros(ns, dtype=cotangent_dtype),
            jnp.zeros(nnu, dtype=cotangent_dtype),
            linear_solver,
        )
    else:
        P_inv_vec, L_H, L_F = factorize_kkt(Q, G, A, s, lam, linear_solver)
        dz, _, dlam_tilde, dnu = solve_kkt_rhs(
            G,
            A,
            s,
            lam,
            P_inv_vec,
            L_H,
            L_F,
            -dl_dz,
            jnp.zeros(ns, dtype=cotangent_dtype),
            jnp.zeros(ns, dtype=cotangent_dtype),
            jnp.zeros(nnu, dtype=cotangent_dtype),
            linear_solver,
        )

    # recover real dlam from our modified (symmetrized) KKT system
    dlam = dlam_tilde / lam

    return optnet_derivatives(dz, dlam, dnu, z, lam, nu)


def _make_solve_qp_primal(
    linear_solver: LinearSolver = LinearSolver.CHOLESKY,
    *,
    full_kkt: bool = False,
):
    @jax.custom_vjp
    def solve_qp_primal_backend(
        Q, q, A, b, G, h, solver_tol=1e-5, target_kappa=1e-3, max_iter=30
    ):
        x, _, _, _, _, _ = solve_qp(
            Q,
            q,
            A,
            b,
            G,
            h,
            solver_tol=solver_tol,
            max_iter=max_iter,
            linear_solver=linear_solver,
            full_kkt=full_kkt,
        )
        return x

    def solve_qp_primal_forward(
        Q, q, A, b, G, h, solver_tol=1e-5, target_kappa=1e-3, max_iter=30
    ):
        x, s, z, y, _, _ = solve_qp(
            Q,
            q,
            A,
            b,
            G,
            h,
            solver_tol=solver_tol,
            max_iter=max_iter,
            linear_solver=linear_solver,
            full_kkt=full_kkt,
        )
        xr, sr, zr, yr, _, _ = relax_qp(
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
            solver_tol=solver_tol,
            target_kappa=target_kappa,
            max_iter=max_iter,
            linear_solver=linear_solver,
            full_kkt=full_kkt,
        )
        res = (Q, q, A, b, G, h, xr, sr, zr, yr)
        return x, res

    def solve_qp_primal_backward(res, input_grad):
        Q, q, A, b, G, h, xr, sr, zr, yr = res
        return (
            *diff_qp(
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
                input_grad,
                linear_solver,
                full_kkt,
            ),
            None,
            None,
            None,
        )

    solve_qp_primal_backend.defvjp(
        solve_qp_primal_forward, solve_qp_primal_backward
    )
    return solve_qp_primal_backend


solve_qp_primal = _make_solve_qp_primal()
solve_qp_primal_qr = _make_solve_qp_primal(LinearSolver.QR)
solve_qp_primal_full_lu = _make_solve_qp_primal(
    LinearSolver.LU, full_kkt=True
)
solve_qp_primal_full_qr = _make_solve_qp_primal(
    LinearSolver.QR, full_kkt=True
)
