"""Tests for the elastic QP backend dispatch."""

import jax.numpy as jnp
import pytest

import qpax


def test_elastic_explicit_smoke():
    Q = jnp.eye(2)
    q = jnp.zeros(2)
    G = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    h = jnp.array([1.0, 1.0])
    penalty = jnp.array(1.0)

    out = qpax.solve_qp_elastic(Q, q, G, h, penalty, backend="e")
    x = out[0]
    assert x.shape == (2,)
    assert jnp.all(jnp.isfinite(x))


def test_elastic_implicit_raises():
    Q = jnp.eye(2)
    q = jnp.zeros(2)
    G = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    h = jnp.array([1.0, 1.0])
    penalty = jnp.array(1.0)

    with pytest.raises(NotImplementedError, match="implicit backend"):
        qpax.solve_qp_elastic(Q, q, G, h, penalty, backend="i")

    with pytest.raises(NotImplementedError, match="implicit backend"):
        qpax.solve_qp_elastic_primal(Q, q, G, h, penalty, backend="i")
