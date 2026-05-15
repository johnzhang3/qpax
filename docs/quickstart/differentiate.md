# Differentiate a QP

`qpax` exposes a separate entry point,
[`qpax.solve_qp_primal`][qpax.solve_qp_primal], that returns only the
primal `x` and carries a custom VJP. That makes it a drop-in differentiable
layer: wrap it in `jax.grad` (or `jax.value_and_grad`) and gradients flow
back into every QP parameter $(Q, q, A, b, G, h)$.

## The problem

The example sets up a small non-negative least-squares problem and casts
it to standard QP form:

$$
\min_{x}\;\tfrac{1}{2}\,\|F x - g\|_2^{2}
\quad \text{s.t.}\quad x \ge 0.
$$

It then defines a tracking loss on the solution,

$$
L(x) \;=\; \|x - \mathbf{1}\|_2^{2},
$$

and computes $\nabla_{Q,q,A,b,G,h}\, L$ with a single
`jax.value_and_grad` call.

## Code

```python
--8<-- "examples/diff_qp.py"
```
