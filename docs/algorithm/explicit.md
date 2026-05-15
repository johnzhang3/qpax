# Explicit algorithm

The explicit backend (`backend="e"`) is the **predictor–corrector** PDIP
from `cvxgen`. Steps in the slack/dual variables are taken directly,
shortened by a fraction-to-the-boundary line search.

## Solving a QP

```text
function solve_qp_explicit(Q, q, A, b, G, h; tol, max_iter)

    (x, s, z, y) ← initialize(Q, q, A, b, G, h)

    for i = 1, ..., max_iter do

        # Residuals and convergence check
        r ← evaluate_residuals(Q, q, A, b, G, h, x, y, z, s)
        if ‖r‖∞ < tol then
            return x, y, z, s
        end if

        # Factor the reduced KKT system once per iteration
        K̃ ← precompute_kkt_factors(Q, A, G, s, z)

        # 1. Predictor: affine step with μ = 0
        (Δxa, Δsa, Δza, Δya) ← solve_kkt(K̃, r, μ = 0)
        αa ← linesearch(s, z, Δsa, Δza)

        # 2. Centering parameter
        μ      ← (sᵀ z) / m
        μ_aff  ← ((s + αa Δsa)ᵀ (z + αa Δza)) / m
        σ      ← (μ_aff / μ)^3

        # 3. Corrector with second-order correction in complementarity
        (Δx, Δs, Δz, Δy) ← solve_kkt(K̃, r, σμ − Δsa ⊙ Δza)

        # 4. Step
        α ← linesearch(s, z, Δs, Δz)
        (x, s, z, y) ← (x + αΔx, s + αΔs, z + αΔz, y + αΔy)

    end for

end function
```

## Relaxing the solution

After convergence, the solution is refined to a $\kappa$-relaxed KKT point
to produce well-conditioned gradients:

```text
function relax_qp_explicit(Q, q, A, b, G, h, x, s, z, y; κ_relax, tol, max_iter)

    for i = 1, ..., max_iter do
        r ← evaluate_residuals(Q, q, A, b, G, h, x, y, z, s)
        if ‖r‖∞ < tol then
            return x, y, z, s
        end if

        K̃ ← precompute_kkt_factors(Q, A, G, s, z)
        (Δx, Δs, Δz, Δy) ← solve_kkt(K̃, r, κ_relax)
        α ← linesearch(s, z, Δs, Δz)

        (x, s, z, y) ← (x + αΔx, s + αΔs, z + αΔz, y + αΔy)
    end for

end function
```

## Computing gradients

The VJP reuses the cached KKT factorization from the forward pass:

```text
function compute_qp_gradients_explicit(Q, q, A, b, G, h, x, y, z, s, κ_relax, ∇xl; K̃)

    r ← (-∇xl, 0, 0, 0)
    (dx, dy, dz, ds) ← solve_kkt(K̃, r, κ_relax)

    ∇Ql ← (dx xᵀ + x dxᵀ) / 2
    ∇ql ← dx
    ∇Al ← dy xᵀ + y dxᵀ
    ∇bl ← -dy
    ∇Gl ← dz xᵀ + z dxᵀ
    ∇hl ← -dz

    return ∇Ql, ∇ql, ∇Al, ∇bl, ∇Gl, ∇hl

end function
```
