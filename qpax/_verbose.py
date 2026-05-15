"""Verbose-mode banner and summary prints shared by all PDIP backends."""


def print_header(n, m, p, tol, max_iter, precision, backend, sigma=None):
    print()
    print("-------------------------------------------------------------")
    print("                            QPAX                             ")
    print("                Quadratic Programming in JAX                ")
    print("-------------------------------------------------------------")
    print("\nproblem:")
    print(f" variables (n): {n}")
    print(f" inequality constraints (p): {p}")
    print(f" equality constraints (m): {m}")
    print("\nsettings:")
    print(f" precision: {precision}")
    print(f" solver tolerance: {float(tol):.2e}")
    print(f" maximum iterations: {max_iter}")
    print(f" backend: {backend}")
    if sigma is not None:
        print(f" sigma: {float(sigma):.4f}")
    print()


def print_footer(converged, cost, iters):
    # print("-------------------------------------------------------------")
    print(
        "----------------------------------------------------------------------------------------"
    )

    print(f"converged: {bool(int(converged))}")
    print(f"cost: {float(cost):.4f}")
    print(f"number of iterations: {int(iters)}\n")
