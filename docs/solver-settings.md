# Solver settings

These are the settings of the `qpax` solver:

| Argument        | Default                 | Backend | Description                                                                                              |
| --------------- | ----------------------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `backend`       | `"i"`        | —       | Selects the explicit (`"e"`) or implicit retraction-manifold (`"i"`) complementarity backend \*.            |
| `solver_tol`    | `1e-5`                  | e, i    | Stopping tolerance on the $L_\infty$ norm of the KKT residual.                                           |
| `max_iter`      | `30`                    | e, i    | Maximum number of iterations when solving the QP.                                                                             |
| `verbose`       | `False`                 | e, i    | If `True`, prints per-iteration residual, step size, and centering info.                                 |
| `linear_solver` | `CHOLESKY` | e       | Factorization for the reduced KKT system (`CHOLESKY` or `QR`). Implicit backend always uses LU.          |
| `sigma`         | `0.125`                 | i       | Centering parameter $\sigma$ targeting the next duality gap. Accepted in `e` for API uniformity but unused (explicit uses Mehrotra centering). |
| `target_kappa`  | `1e-3`        | e, i    | Relaxation parameter $\kappa$ for the perturbed complementarity condition $s \cdot z = \kappa$. Used only by `relax_qp` and `solve_qp_primal`. |

\* For further details on boths backends, see [Algorithm]() (comming soon).


