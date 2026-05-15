"""Top-level PDIP dispatcher.

Routes ``solve_qp`` calls to either the explicit (``backend="e"``) or
implicit (``backend="i"``) backend. Shared data types (``LinearSolver``,
``QPData``, ``QPState``, ``SolverParams``) are re-exported from the explicit
backend; both backends define structurally identical versions.
"""

from typing import Any

import jax

from qpax.explicit import pdip as _explicit
from qpax.explicit.pdip import LinearSolver, QPData, QPState, SolverParams
from qpax.implicit import pdip as _implicit


def solve_qp(
    Q: jax.Array,
    q: jax.Array,
    A: jax.Array,
    b: jax.Array,
    G: jax.Array,
    h: jax.Array,
    *,
    backend: str = "i",
    **kwargs: Any,
) -> tuple:
    """Solve a convex QP via primal-dual interior point.

    Args:
        Q: ``(n, n)`` PSD cost matrix.
        q: ``(n,)`` linear cost.
        A: ``(m, n)`` equality constraint matrix.
        b: ``(m,)`` equality constraint RHS.
        G: ``(p, n)`` inequality constraint matrix.
        h: ``(p,)`` inequality constraint RHS.
        backend: ``"e"`` for the explicit predictor-corrector PDIP (default),
            ``"i"`` for the implicit retraction-manifold PDIP.
        **kwargs: forwarded to the selected backend (e.g. ``solver_tol``,
            ``max_iter``, ``linear_solver``).

    Returns:
        Tuple ``(x, s, z, y, converged, iters)``.
    """
    if backend == "e":
        return _explicit.solve_qp(Q, q, A, b, G, h, **kwargs)
    if backend == "i":
        return _implicit.solve_qp(Q, q, A, b, G, h, **kwargs)
    raise ValueError(f"unknown backend {backend!r}; expected 'e' or 'i'")


__all__ = [
    "LinearSolver",
    "QPData",
    "QPState",
    "SolverParams",
    "solve_qp",
]
