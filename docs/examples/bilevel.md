# Bilevel optimization

Bilevel problems have an *outer* objective evaluated at the solution of an
*inner* QP. Differentiability lets the outer optimizer take gradient steps
through the inner solve, which is the textbook use case for `qpax`.

The pattern is: build the inner QP from outer parameters, solve it with
`solve_qp_primal`, plug the result into the outer loss, and let `jax.grad`
backpropagate through the whole pipeline.

```python
import qpax  # todo: bilevel optimization example
```
