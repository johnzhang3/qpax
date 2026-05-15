"""Explicit (predictor-corrector) PDIP backend for qpax.

Algorithmic bodies in this subpackage are kept distinct from the implicit
backend so the two can be read and maintained independently. The public API
is exposed through ``qpax`` via the top-level dispatchers; import directly
from this subpackage only when you specifically want the explicit backend.
"""
