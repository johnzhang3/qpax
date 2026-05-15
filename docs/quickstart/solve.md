# Solve a QP

The simplest thing `qpax` does: take the data of a convex quadratic program
and return its primal solution along with convergence diagnostics. Reach
for this recipe when you have a single QP and just need the answer.

## The problem

The example builds a random feasible convex QP with $n=3$ variables, $m=1$
equality constraint and $p=2$ inequality constraints, then solves it:

$$
\begin{aligned}
\min_{x}\;& \tfrac{1}{2}\, x^{\mathsf T} Q x + q^{\mathsf T} x \\
\text{s.t.}\;& A x = b \\
& G x \le h.
\end{aligned}
$$

[`qpax.solve_qp`][qpax.solve_qp] returns the tuple
`(x, s, z, y, converged, iters)`: the primal `x`, the inequality slack `s`
and dual `z`, the equality dual `y`, a convergence flag, and the iteration
count.

## Code

```python
--8<-- "examples/solve_qp.py"
```
