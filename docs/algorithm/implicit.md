# Implicit algorithm

The implicit backend (`backend="i"`) parameterises the slack/dual pair by
manifold coordinates $v = z - s$ and a scalar gap $\kappa = (s^{\top} z) / m$,
then **retracts** back to the positive cone after each step. The Newton step
is computed against a reduced *implicit* KKT matrix that is intrinsically
positive-definite.

The Newton step uses the helper below to recover
$(\Delta x, \Delta s, \Delta z, \Delta y)$ from a single reduced solve:

```text
function solve_implicit_kkt_rhs(G, B_n, B_p, c, L_J, r_t, r_e, r_i, r_z, r_s, r_k)

    # L_J stores LU factors of the reduced implicit KKT matrix
    #
    #     J = [ Q - GᵀG    Gᵀ    Aᵀ ]
    #         [ G          -B_n   0  ]
    #         [ A           0     0  ]

    r ← (r_t - Gᵀ(r_i + r_z - r_s), r_i - r_s - c·r_k, r_e)
    (Δx, Δv, Δy) ← lu_solve(L_J, -r)

    Δz ←  -r_z + vec(B_p) ⊙ Δv - c·r_k
    Δs ←  -r_s - vec(B_n) ⊙ Δv - c·r_k

    return Δx, Δs, Δz, Δy

end function
```

## Solving a QP

```text
function solve_qp_implicit(Q, q, A, b, G, h; σ, tol, max_iter)

    (x, s, z, y) ← initialize(Q, q, A, b, G, h)

    for i = 1, ..., max_iter do

        # Manifold coordinates
        v ← z - s
        κ ← (sᵀ z) / m

        r ← evaluate_residuals(Q, q, A, b, G, h, x, y, z, s)
        if ‖r‖∞ < tol then
            return x, y, z, s
        end if

        K̃        ← precompute_kkt_factors(Q, A, G, v, κ)
        κ_target ← σκ
        (Δx, Δy, Δz, Δs, Δv, Δκ) ← solve_kkt(K̃, r, κ_target)

        α ← linesearch(s, z, Δs, Δz)

        x ← x + αΔx
        y ← y + αΔy
        v ← v + αΔv
        κ ← κ + αΔκ

        # Retract back to positive slack-dual variables
        (z, s) ← retraction_map(v, κ)

    end for

end function
```

## Relaxing the solution

```text
function relax_qp_implicit(Q, q, A, b, G, h, x, y, z, s, κ_relax; K̃, tol, max_iter)

    for i = 1, ..., max_iter do
        v ← z - s
        κ ← (sᵀ z) / m

        r ← evaluate_residuals(Q, q, A, b, G, h, x, y, z, s)
        if ‖r‖∞ < tol then
            return x, y, z, s
        end if

        if K̃ is not cached then
            K̃ ← precompute_kkt_factors(Q, A, G, v, κ)
        end if

        (Δx, Δy, Δz, Δs, Δv, Δκ) ← solve_kkt(K̃, r, κ_relax)
        α ← linesearch(s, z, Δs, Δz)

        x ← x + αΔx
        y ← y + αΔy
        v ← v + αΔv
        κ ← κ + αΔκ

        (z, s) ← retraction_map(v, κ)
    end for

end function
```

## Computing gradients

```text
function compute_qp_gradients_implicit(Q, q, A, b, G, h, x, y, z, s, κ_relax, ∇xl; K̃)

    r ← (-∇xl, 0, 0, 0, 0)
    (dx, dy, dz, ds, _, _) ← solve_kkt(K̃, r, κ_relax)

    ∇Ql ← (dx xᵀ + x dxᵀ) / 2
    ∇ql ← dx
    ∇Al ← dy xᵀ + y dxᵀ
    ∇bl ← -dy
    ∇Gl ← dz xᵀ + z dxᵀ
    ∇hl ← -dz

    return ∇Ql, ∇ql, ∇Al, ∇bl, ∇Gl, ∇hl

end function
```
