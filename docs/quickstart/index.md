# Quickstart

Minimal recipes for each thing `qpax` can do. Every code block on the
following pages is the verbatim contents of a runnable script under
[`examples/`](https://github.com/qpax-solver/qpax/tree/main/examples),
so you can copy a file out of the repo and run it as-is. Pick the recipe
you need:

- **[Solve a QP](solve.md)** — single forward solve.
- **[Differentiate a QP](differentiate.md)** — pass gradients through the
  solution with `jax.grad`.
- **[Batched solving](batched-solve.md)** — solve many independent QPs in
  parallel with `jax.vmap`.
- **[Batched differentiating](batched-differentiate.md)** — gradients
  through a whole batch at once.
