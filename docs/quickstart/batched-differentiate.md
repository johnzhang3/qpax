# Batched differentiating

`solve_qp_primal` is fully `vmap`-compatible, so the batched-solve pattern
extends to gradients with no extra plumbing: stack
`jit(vmap(value_and_grad(...)))` and you get a batch of losses *and* a
batch of gradients in one fused call.

## The problem

The example draws $N = 256$ random non-negative least-squares problems,
cast to standard QP form, and defines a per-problem tracking loss:

$$
\min_{x_i}\;\tfrac{1}{2}\,\|F_i x_i - g_i\|_2^{2}
\quad \text{s.t.}\quad x_i \ge 0,
\qquad
L_i(x_i) \;=\; \|x_i - \mathbf{1}\|_2^{2}.
$$

A single batched call then evaluates every $L_i$ and every
$\nabla_{Q_i, q_i, A_i, b_i, G_i, h_i}\, L_i$ in parallel.

## Code

```python
--8<-- "examples/batch_diff_qp.py"
```
