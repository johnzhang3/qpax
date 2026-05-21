"""Top-level relaxed-PDIP dispatcher.

Routes ``relax_qp`` calls to either the explicit (``backend="e"``) or
implicit (``backend="i"``) backend.
"""

from typing import Any

import jax

from qpax.explicit import pdip_relaxed as _explicit
from qpax.explicit.pdip import LinearSolver
from qpax.implicit import pdip_relaxed as _implicit


def relax_qp(
    Q: jax.Array,
    q: jax.Array,
    A: jax.Array,
    b: jax.Array,
    G: jax.Array,
    h: jax.Array,
    x: jax.Array,
    s: jax.Array,
    z: jax.Array,
    y: jax.Array,
    *,
    backend: str = "e",
    **kwargs: Any,
) -> tuple:
    """Refine a PDIP solution to a relaxed (kappa-perturbed) KKT point.

    Args:
        Q: ``(n, n)`` PSD cost matrix.
        q: ``(n,)`` linear cost.
        A: ``(m, n)`` equality constraint matrix.
        b: ``(m,)`` equality constraint RHS.
        G: ``(p, n)`` inequality constraint matrix.
        h: ``(p,)`` inequality constraint RHS.
        x: primal warm start, shape ``(n,)``.
        s: inequality slack warm start, shape ``(p,)``.
        z: inequality dual warm start, shape ``(p,)``.
        y: equality dual warm start, shape ``(m,)``.
        backend: ``"e"`` for the explicit backend (default), ``"i"`` for
            the implicit backend.
        **kwargs: forwarded to the selected backend (e.g. ``solver_tol``,
            ``target_kappa``, ``max_iter``).

    Returns:
        Tuple ``(x, s, z, y, converged, iters)`` of the refined solution.
    """
    if backend == "e":
        return _explicit.relax_qp(Q, q, A, b, G, h, x, s, z, y, **kwargs)
    if backend == "e_qr":
        return _explicit.relax_qp(
            Q, q, A, b, G, h, x, s, z, y,
            linear_solver=LinearSolver.QR,
            **kwargs,
        )
    if backend == "e_full_lu":
        return _explicit.relax_qp(
            Q, q, A, b, G, h, x, s, z, y,
            linear_solver=LinearSolver.LU,
            full_kkt=True,
            **kwargs,
        )
    if backend == "e_full_qr":
        return _explicit.relax_qp(
            Q, q, A, b, G, h, x, s, z, y,
            linear_solver=LinearSolver.QR,
            full_kkt=True,
            **kwargs,
        )
    if backend == "i":
        xr, sr, zr, yr, _, _, _, _, converged, iters = _implicit.relax_qp(
            Q, q, A, b, G, h, x, s, z, y, **kwargs
        )
        return xr, sr, zr, yr, converged, iters
    raise ValueError(
        f"unknown backend {backend!r}; expected 'e', 'e_qr', "
        "'e_full_lu', 'e_full_qr', or 'i'"
    )


__all__ = ["relax_qp"]
