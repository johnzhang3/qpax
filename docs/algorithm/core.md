# Core algorithm

Every QP `qpax` solves has the form

$$
\underset{x}{\text{minimize}} \quad \tfrac{1}{2}\,x^{\top} Q x + q^{\top} x
\quad \text{s.t.} \quad
A x = b, \;\; G x \leq h, \quad Q \succeq 0.
$$

Introducing a slack $s \geq 0$, equality multiplier $y$, and inequality
multiplier $z \geq 0$, the KKT conditions are

$$
\begin{aligned}
Q x + q + A^{\top} y + G^{\top} z &= 0, \\
G x + s - h &= 0, \\
A x - b &= 0, \\
s_i z_i &= 0, \qquad s, z \geq 0.
\end{aligned}
$$

The PDIP idea is to replace the hard complementarity $s_i z_i = 0$ with a
smooth condition $s_i z_i = \mu$, then drive $\mu \to 0$. Each iteration:

1. **Form residuals.** Evaluate the KKT residual at the current iterate.
2. **Factor the KKT system.** Reduce the saddle-point system to a smaller
   positive-definite system in $x$ using the elimination from `cvxopt`, and
   factor it (Cholesky by default, QR on request).
3. **Compute a Newton step.** Solve the reduced KKT system for the step
   directions $(\Delta x, \Delta s, \Delta z, \Delta y)$.
4. **Choose a step size.** Pick $\alpha \in (0, 1]$ so that the slack/dual
   variables stay strictly positive.
5. **Update.** Take a step of length $\alpha$ along the Newton direction.

This skeleton is shared by both backends. They differ in *how* the
slack/dual variables are updated (step 5) and, as a consequence, in how the
relaxed KKT point is reached and differentiated.
