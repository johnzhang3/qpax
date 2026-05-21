import jax
import jax.numpy as jnp

from qpax._verbose import print_footer, print_header
from qpax.explicit.pdip import (
    LinearSolver,
    _all_finite,
    factorize_kkt,
    factorize_full_kkt,
    ort_linesearch,
    solve_full_kkt_rhs,
    solve_kkt_rhs,
)


def pdip_newton_step(
    inputs,
    verbose: bool = False,
    linear_solver: LinearSolver = LinearSolver.CHOLESKY,
    full_kkt: bool = False,
):
    """
    Algorithm 3 Relaxing a Quadratic Program
    """

    # unpack inputs
    (
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
        solver_tol,
        converged,
        pdip_iter,
        target_kappa,
        bad_step_seen,
    ) = inputs

    # evaluate relaxed KKT conditions
    r1 = Q @ x + q + A.T @ y + G.T @ z
    r2 = s * z - target_kappa  # we added this (relaxed complementarity)
    r3 = G @ x + s - h
    r4 = A @ x - b

    # check convergence
    kkt_res = jnp.concatenate((r1, r2, r3, r4))
    converged = jnp.where(jnp.linalg.norm(kkt_res, ord=jnp.inf) < solver_tol, 1, 0)

    # calculate and take Newton step
    if full_kkt:
        factor_info = factorize_full_kkt(Q, G, A, s, z, linear_solver)
        dx, ds, dz, dy = solve_full_kkt_rhs(
            G, A, factor_info, -r1, -r2, -r3, -r4, linear_solver
        )
    else:
        factor_info, L_H, L_F = factorize_kkt(Q, G, A, s, z, linear_solver)
        dx, ds, dz, dy = solve_kkt_rhs(
            G, A, s, z, factor_info, L_H, L_F, -r1, -r2, -r3, -r4,
            linear_solver,
        )

    # linesearch and update primal & dual vars
    alpha = 0.99 * jnp.min(
        jnp.array([1.0, 0.99 * ort_linesearch(s, ds), 0.99 * ort_linesearch(z, dz)])
    )
    step_finite = _all_finite(factor_info, dx, ds, dz, dy, alpha)
    bad_step_seen = jnp.logical_or(bad_step_seen, jnp.logical_not(step_finite))

    if verbose:
        r4_print = r4 if len(r4) > 0 else jnp.zeros(1)
        print(
            f"{pdip_iter:3d}   "
            f"{jnp.linalg.norm(r1, ord=jnp.inf):9.2e}   "
            f"{jnp.linalg.norm(r2, ord=jnp.inf):9.2e}   "
            f"{jnp.linalg.norm(r3, ord=jnp.inf):9.2e}   "
            f"{jnp.linalg.norm(r4_print, ord=jnp.inf):9.2e}   "
            f"{alpha:6.4f}   {target_kappa:9.2e}   {bool(step_finite)}"
        )

    # Under vmap, jax.lax.while_loop runs every lane until the slowest one
    # converges; freezing already-converged lanes prevents post-convergence
    # Newton-step noise from drifting the state into NaN in f32.
    take = converged == 0
    x = jnp.where(take, x + alpha * dx, x)
    s = jnp.where(take, s + alpha * ds, s)
    z = jnp.where(take, z + alpha * dz, z)
    y = jnp.where(take, y + alpha * dy, y)

    return (
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
        solver_tol,
        converged,
        pdip_iter + 1,
        target_kappa,
        bad_step_seen,
    )


# 0 Q
# 1 q
# 2 A
# 3 b
# 4 G
# 5 h
# 6 x
# 7 s
# 8 z
# 9 y
# 10 solver_tol
# 11 converged
# 12 pdip_iter


def relax_qp(
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
    solver_tol=1e-5,
    target_kappa=1e-3,
    max_iter=30,
    linear_solver: LinearSolver = LinearSolver.CHOLESKY,
    full_kkt: bool = False,
    return_bad_step=False,
    sigma: float = 0.125,  # noqa: ARG001 — accepted for API uniformity; explicit uses Mehrotra centering
    verbose: bool = False,
):
    # Floor the warm-start so Cholesky of H = Q + Gᵀ diag(z/s) G stays
    # well-conditioned in f32. A tight PDIP forward drives s,z below ε_f32
    # and max(z/s) past 1e7, beyond f32 Cholesky's reach. Flooring at √κ
    # caps cond(H) while Newton iterations restore primal-dual feasibility.
    floor = jnp.sqrt(target_kappa)
    s = jnp.maximum(s, floor)
    z = jnp.maximum(z, floor)

    # continuation criteria for normal predictor-corrector
    def relaxed_continuation_criteria(inputs):
        converged = inputs[11]
        pdip_iter = inputs[12]

        return jnp.logical_and(pdip_iter < max_iter, converged == 0)

    converged = 0
    pdip_iter = 0
    init_inputs = (
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
        solver_tol,
        converged,
        pdip_iter,
        target_kappa,
        jnp.bool_(False),
    )

    if verbose:
        print_header(
            n=Q.shape[0],
            m=A.shape[0],
            p=G.shape[0],
            tol=solver_tol,
            max_iter=max_iter,
            precision="f32" if Q.dtype == jnp.float32 else "f64",
            backend="explicit-full-kkt" if full_kkt else "explicit",
        )
        print(
            "iter      r1          r2         r3         r4         alpha      kappa     finite"  # noqa: E501
        )
        print(
            "------------------------------------------------------------------------------------------------"
        )
        outputs = init_inputs
        while relaxed_continuation_criteria(outputs):
            outputs = pdip_newton_step(
                outputs,
                verbose=True,
                linear_solver=linear_solver,
                full_kkt=full_kkt,
            )
    else:
        def step_fn(inputs):
            return pdip_newton_step(
                inputs,
                linear_solver=linear_solver,
                full_kkt=full_kkt,
            )

        outputs = jax.lax.while_loop(
            relaxed_continuation_criteria, step_fn, init_inputs
        )

    x_rlx, s_rlx, z_rlx, y_rlx = outputs[6:10]
    converged = outputs[11]
    pdip_iter = outputs[12]
    bad_step_seen = outputs[14]

    if verbose:
        print_footer(converged, 0.5 * x_rlx @ Q @ x_rlx + q @ x_rlx, pdip_iter)

    results = (x_rlx, s_rlx, z_rlx, y_rlx, converged, pdip_iter)
    return (*results, bad_step_seen) if return_bad_step else results
