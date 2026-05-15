"""Top-level differentiable-QP dispatcher.

Routes ``solve_qp_primal`` to either the explicit (``backend="e"``) or
implicit (``backend="i"``) backend. Each backend defines its own
``jax.custom_vjp`` for reverse-mode differentiation; the dispatcher
branches on ``backend`` at Python level and lets JAX trace into the
chosen backend's custom-VJP primitive.
"""

from typing import Any

import jax

from qpax.explicit import diff_qp as _explicit
from qpax.implicit import diff_qp as _implicit


def solve_qp_primal(
    Q: jax.Array,
    q: jax.Array,
    A: jax.Array,
    b: jax.Array,
    G: jax.Array,
    h: jax.Array,
    *,
    backend: str = "i",
    **kwargs: Any,
) -> jax.Array:
    """Solve a QP and return the primal solution ``x``.

    Differentiable via ``jax.custom_vjp``; gradients flow through every QP
    parameter using the implicit function theorem on the relaxed KKT system.

    Args:
        Q: ``(n, n)`` PSD cost matrix.
        q: ``(n,)`` linear cost.
        A: ``(m, n)`` equality constraint matrix.
        b: ``(m,)`` equality constraint RHS.
        G: ``(p, n)`` inequality constraint matrix.
        h: ``(p,)`` inequality constraint RHS.
        backend: ``"e"`` for the explicit backend (default), ``"i"`` for
            the implicit backend.
        **kwargs: forwarded to the selected backend (e.g. ``solver_tol``,
            ``target_kappa``, ``max_iter``).

    Returns:
        Primal solution ``x`` of shape ``(n,)``.
    """
    if backend == "e":
        return _explicit.solve_qp_primal(Q, q, A, b, G, h, **kwargs)
    if backend == "i":
        return _implicit.solve_qp_primal(Q, q, A, b, G, h, **kwargs)
    raise ValueError(f"unknown backend {backend!r}; expected 'e' or 'i'")


__all__ = ["solve_qp_primal"]
