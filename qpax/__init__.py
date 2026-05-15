"""qpax — differentiable QP solver in JAX with selectable backends.

Use the ``backend`` keyword argument to switch between the explicit
predictor-corrector PDIP (``backend="e"``, default) and the implicit
retraction-manifold PDIP (``backend="i"``). The two algorithmic
implementations live in ``qpax.explicit`` and ``qpax.implicit``; this
package's top-level modules are thin dispatchers over those.
"""

from qpax.diff_qp import solve_qp_primal
from qpax.elastic_qp import solve_qp_elastic, solve_qp_elastic_primal
from qpax.pdip import LinearSolver, QPData, QPState, SolverParams, solve_qp
from qpax.pdip_relaxed import relax_qp

__all__ = [
    "LinearSolver",
    "QPData",
    "QPState",
    "SolverParams",
    "relax_qp",
    "solve_qp",
    "solve_qp_elastic",
    "solve_qp_elastic_primal",
    "solve_qp_primal",
]
