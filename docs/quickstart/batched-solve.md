# Batched solving

When you have many independent QPs of the same shape, you almost never
want to loop over them in Python. Wrap the solver in `jax.vmap` and fuse
the whole batch into a single accelerator call; add `jax.jit` to amortise
the trace/compile cost over many batches.

## The problem

The example draws $N = 10{,}000$ random non-negative least-squares
problems and solves all of them in parallel. Each one is

$$
\min_{x_i}\;\tfrac{1}{2}\,\|F_i x_i - g_i\|_2^{2}
\quad \text{s.t.}\quad x_i \ge 0,
\qquad i = 1, \dots, N,
$$

cast to standard QP form. `vmap` is applied twice: once to convert the
batch of $(F_i, g_i)$ to batched QP data, and once to solve the batched
QPs.

## Code

```python
--8<-- "examples/batch_solve_qp.py"
```
