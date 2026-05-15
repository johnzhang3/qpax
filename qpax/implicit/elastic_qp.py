"""Implicit-backend elastic QP — not yet implemented.

The elastic QP formulation only has an explicit backend at the moment. These
stubs mirror the explicit API so callers can switch via ``backend="i"`` and
get an informative error instead of an ``AttributeError``.
"""

_MSG = (
    "Elastic QP is not implemented for the implicit backend (backend='i'). "
    "Use backend='e' (explicit) for elastic problems."
)


def solve_qp_elastic(*args, **kwargs):
    raise NotImplementedError(_MSG)


def solve_qp_elastic_primal(*args, **kwargs):
    raise NotImplementedError(_MSG)
