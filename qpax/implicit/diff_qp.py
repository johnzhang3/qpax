import jax
import jax.numpy as jnp

from .pdip import solve_implicit_kkt_rhs, solve_qp
from .pdip_relaxed import relax_qp


def implicit_derivatives(dx, dy, dg, x, z, y):

    dl_dQ = 0.5 * (jnp.outer(dx, x) + jnp.outer(x, dx))
    dl_dA = jnp.outer(dy, x) + jnp.outer(y, dx)
    dl_dG = jnp.outer(dg, x) + jnp.outer(z, dx)

    dl_dq = dx
    dl_db = -dy
    dl_dh = -dg

    return dl_dQ, dl_dq, dl_dA, dl_db, dl_dG, dl_dh


def diff_qp(b, G, h, x, z, y, kappa, Bp_vec, Bn_vec, c_vec, L_J, dl_dx):

    dx, _, dg, dy, _, _ = solve_implicit_kkt_rhs(
        G,
        Bn_vec,
        Bp_vec,
        c_vec,
        L_J,
        dl_dx,
        jnp.zeros_like(b),
        jnp.zeros_like(h),
        jnp.zeros_like(h),
        jnp.zeros_like(h),
        jnp.zeros_like(kappa),
    )

    return implicit_derivatives(dx, dy, dg, x, z, y)


@jax.custom_vjp
def solve_qp_primal(Q, q, A, b, G, h, solver_tol=1e-5, target_kappa=1e-3, max_iter=30):

    x, s, z, y, _, _ = solve_qp(
        Q, q, A, b, G, h, solver_tol=solver_tol, max_iter=max_iter
    )
    return x


"""
these two functions are only called when we diff solve_qp_x
"""


def solve_qp_primal_forward(
    Q, q, A, b, G, h, solver_tol=1e-5, target_kappa=1e-3, max_iter=30
):

    x, s, z, y, _, _ = solve_qp(
        Q, q, A, b, G, h, solver_tol=solver_tol, max_iter=max_iter
    )
    xr, sr, zr, yr, Bp_vec, Bn_vec, c_vec, L_J, _, _ = relax_qp(
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
    )
    res = (b, G, h, xr, zr, yr, target_kappa, Bp_vec, Bn_vec, c_vec, L_J)
    return x, res


def solve_qp_primal_backward(res, input_grad):

    b, G, h, xr, zr, yr, kappa, Bp_vec, Bn_vec, c_vec, L_J = res

    return (
        *diff_qp(b, G, h, xr, zr, yr, kappa, Bp_vec, Bn_vec, c_vec, L_J, input_grad),
        None,
        None,
        None,
    )


solve_qp_primal.defvjp(solve_qp_primal_forward, solve_qp_primal_backward)
