---
hide:
  - toc
---

<style>
  .md-content__inner > h1:first-of-type { display: none; }
</style>

<p align="center">
  <img src="assets/images/logo_nobackground_cropped.png" alt="qpax logo" width="640">
</p>

<p align="center"><strong>Differentiable, batched, single-precision quadratic programming in JAX</strong></p>

<p align="center">
  <a href="installation/" class="md-button md-button--primary">Get started</a>
  <a href="https://github.com/qpax-solver/qpax" class="md-button">View on GitHub</a>
</p>

---

`qpax` solves and differentiates (batched) convex quadratic programs of the form

$$
\begin{aligned}
\underset{x}{\text{minimize}} \quad & \tfrac{1}{2}\,x^{\top} Q\,x + q^{\top} x \\
\text{subject to} \quad & A x = b, \\
                        & G x \leq h,
\end{aligned}
$$

with decision variables $x \in \mathbb{R}^{n}$ and data matrices $Q \succeq 0$,
$q \in \mathbb{R}^{n}$, $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^{m}$,
$G \in \mathbb{R}^{p \times n}$ and $h \in \mathbb{R}^{p}$.

<p class="eyebrow">Features</p>

<div class="grid cards" markdown>

-   __Differentiable__

    ---

    Backpropagate through QPs and obtain smooth, informative subgradients
    even at active inequality constraints.

-   __Single precision__

    ---

    Runs in `f32`, enabling larger batch sizes and higher throughput on GPU.

-   __Batchable__

    ---

    Solves and differentiates many QPs in parallel with shared structure
    via `jax.vmap`.

-   __Infeasibility avoidance__

    ---

    Avoids generating infeasible problems by solving an always-feasible
    *elastic* QP that returns informative gradients toward feasibility.

</div>
