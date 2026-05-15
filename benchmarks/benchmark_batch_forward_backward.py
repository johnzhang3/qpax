"""Benchmark forward and backward passes across batch sizes.

Forward pass  = `solve_qp` followed by `relax_qp`.
Backward pass = gradient through `solve_qp_primal`.

Backends:
  * "e" — explicit predictor-corrector PDIP
  * "i" — implicit retraction-manifold PDIP
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qpax.explicit import diff_qp as e_diff_qp
from qpax.explicit import pdip as e_pdip
from qpax.explicit import pdip_relaxed as e_pdip_relaxed
from qpax.implicit import diff_qp as i_diff_qp
from qpax.implicit import pdip as i_pdip
from qpax.implicit import pdip_relaxed as i_pdip_relaxed

# ================================ Config ================================== #
NX = 32
NCON = 64
NEQ = 0

BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

N_WARMUP = 1
N_REPEATS = 3

SOLVER_TOL = 1e-5
TARGET_KAPPA = 1e-3
MAX_ITER = 100
PRNG_KEY = 7

INIT_POINT = 0.5
G_MIN = -4.0
G_MAX = 4.0
SLACK_MIN = 0.5
SLACK_MAX = 1.5

RUNS = (
    {"name": "e64", "backend": "e", "use_f64": True, "enabled": 1},
    {"name": "e32", "backend": "e", "use_f64": False, "enabled": 1},
    {"name": "i64", "backend": "i", "use_f64": True, "enabled": 1},
    {"name": "i32", "backend": "i", "use_f64": False, "enabled": 1},
)

SAVE_DATA = True
SAVE_FIGURE = True
DATA_FILENAME = "benchmark_batch_forward_backward.npz"
FIGURE_FILENAME = "benchmark_batch_forward_backward.png"


BACKENDS = {
    "e": (e_pdip.solve_qp, e_pdip_relaxed.relax_qp, e_diff_qp.solve_qp_primal),
    "i": (i_pdip.solve_qp, i_pdip_relaxed.relax_qp, i_diff_qp.solve_qp_primal),
}


def _dtype(use_f64: bool):
    jax.config.update("jax_enable_x64", bool(use_f64))
    return jnp.float64 if use_f64 else jnp.float32


def _make_problem_bank(max_batch: int, dtype):
    key = jax.random.PRNGKey(PRNG_KEY)
    keys = iter(jax.random.split(key, 8))

    def normal(shape):
        return jax.random.normal(next(keys), shape, dtype=dtype)

    def uniform(shape, minval, maxval):
        return jax.random.uniform(
            next(keys), shape, minval=minval, maxval=maxval, dtype=dtype
        )

    Q = jnp.broadcast_to(jnp.eye(NX, dtype=dtype), (max_batch, NX, NX))
    A = jnp.zeros((max_batch, NEQ, NX), dtype=dtype)
    b = jnp.zeros((max_batch, NEQ), dtype=dtype)

    q = -normal((max_batch, NX))
    z0 = jnp.full((max_batch, NX), INIT_POINT, dtype=dtype)
    G = uniform((max_batch, NCON, NX), minval=G_MIN, maxval=G_MAX)
    s0 = uniform((max_batch, NCON), minval=SLACK_MIN, maxval=SLACK_MAX)
    h = jnp.einsum("bij,bj->bi", G, z0) + s0

    return {"Q": Q, "q": q, "A": A, "b": b, "G": G, "h": h}


def _slice_problem(bank, batch_size: int):
    return {k: v[:batch_size] for k, v in bank.items()}


def _build_forward_fn(backend: str):
    solve_qp, relax_qp, _ = BACKENDS[backend]

    def forward_one(Q, q, A, b, G, h):
        x, s, z, y, conv1, iters1 = solve_qp(
            Q, q, A, b, G, h, solver_tol=SOLVER_TOL, max_iter=MAX_ITER
        )
        relax_out = relax_qp(
            Q,
            q,
            A,
            b,
            G,
            h,
            x,
            s,
            z,
            y,
            solver_tol=SOLVER_TOL,
            target_kappa=TARGET_KAPPA,
            max_iter=MAX_ITER,
        )
        conv2, iters2 = relax_out[-2:]
        return conv1, iters1, conv2, iters2

    return jax.jit(jax.vmap(forward_one))


def _build_backward_fn(backend: str):
    _, _, solve_qp_primal = BACKENDS[backend]

    def loss_fn(Q, q, A, b, G, h):
        x = jax.vmap(
            lambda Q, q, A, b, G, h: solve_qp_primal(
                Q,
                q,
                A,
                b,
                G,
                h,
                solver_tol=SOLVER_TOL,
                target_kappa=TARGET_KAPPA,
                max_iter=MAX_ITER,
            )
        )(Q, q, A, b, G, h)
        return jnp.sum(x)

    return jax.jit(jax.grad(loss_fn, argnums=1))


def _time_call(fn, *args):
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return 1000.0 * (time.perf_counter() - t0), out


def _empty_results(n_batch_sizes: int):
    return {
        "forward_ms": np.zeros(n_batch_sizes, dtype=np.float64),
        "backward_ms": np.zeros(n_batch_sizes, dtype=np.float64),
        "median_total_iters": np.zeros(n_batch_sizes, dtype=np.float64),
        "max_total_iters": np.zeros(n_batch_sizes, dtype=np.int32),
        "converged_frac": np.zeros(n_batch_sizes, dtype=np.float64),
    }


def _benchmark_run(run_cfg, batch_sizes):
    dtype = _dtype(bool(run_cfg["use_f64"]))
    problem_bank = _make_problem_bank(int(np.max(batch_sizes)), dtype)
    forward_fn = _build_forward_fn(run_cfg["backend"])
    backward_fn = _build_backward_fn(run_cfg["backend"])
    results = _empty_results(len(batch_sizes))

    print(f"=== {run_cfg['name']} (device={jax.devices()[0]}, dtype={dtype}) ===")
    for i, batch_size in enumerate(batch_sizes):
        bs = int(batch_size)
        batch = _slice_problem(problem_bank, bs)
        args = (batch["Q"], batch["q"], batch["A"], batch["b"], batch["G"], batch["h"])

        for _ in range(N_WARMUP):
            jax.block_until_ready(forward_fn(*args))
            jax.block_until_ready(backward_fn(*args))

        forward_samples, backward_samples = [], []
        forward_out = None
        for _ in range(N_REPEATS):
            sample_ms, forward_out = _time_call(forward_fn, *args)
            forward_samples.append(sample_ms)
            sample_ms, _ = _time_call(backward_fn, *args)
            backward_samples.append(sample_ms)

        solve_conv, solve_iters, relax_conv, relax_iters = forward_out
        total_iters = np.asarray(solve_iters + relax_iters)
        converged = np.asarray(
            jnp.logical_and(solve_conv.astype(bool), relax_conv.astype(bool)),
            dtype=np.float64,
        )

        results["forward_ms"][i] = float(np.median(forward_samples))
        results["backward_ms"][i] = float(np.median(backward_samples))
        results["median_total_iters"][i] = float(np.median(total_iters))
        results["max_total_iters"][i] = int(np.max(total_iters))
        results["converged_frac"][i] = float(np.mean(converged))

        print(
            f"  batch={bs:4d} "
            f"forward_ms={results['forward_ms'][i]:8.3f} "
            f"backward_ms={results['backward_ms'][i]:8.3f} "
            f"median_total_iters={results['median_total_iters'][i]:6.1f} "
            f"max_total_iters={results['max_total_iters'][i]:4d} "
            f"conv={100.0 * results['converged_frac'][i]:6.1f}%"
        )

    return results


def _plot_style(run_cfg):
    color = "#1f77b4" if run_cfg["backend"] == "e" else "#ff7f0e"
    linestyle = "-" if run_cfg["use_f64"] else "-."
    return color, linestyle, 2.0


def _plot(batch_sizes, results_by_name, runs, out_path):
    fig = plt.figure(figsize=(12, 10))
    left_fig, right_fig = fig.subfigures(1, 2, wspace=0.08)
    left_axes = left_fig.subplots(2, 1, sharex=True)
    right_axes = right_fig.subplots(3, 1, sharex=True)

    left_axes[0].set_title("Forward / Backward Wall Time")
    right_axes[0].set_title("Forward Solve + Relax Metrics")

    left_series = (("forward_ms", 1.0), ("backward_ms", 1.0))
    right_series = (
        ("median_total_iters", 1.0),
        ("max_total_iters", 1.0),
        ("converged_frac", 100.0),
    )

    for run_cfg in runs:
        name = run_cfg["name"]
        if name not in results_by_name:
            continue
        results = results_by_name[name]
        color, linestyle, linewidth = _plot_style(run_cfg)

        for ax, (key, scale) in zip(left_axes, left_series, strict=False):
            ax.plot(
                batch_sizes,
                scale * results[key],
                marker="o",
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=name,
            )
        for ax, (key, scale) in zip(right_axes, right_series, strict=False):
            ax.plot(
                batch_sizes,
                scale * results[key],
                marker="o",
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=name,
            )

    left_axes[0].set_ylabel("forward [ms]")
    left_axes[1].set_ylabel("backward [ms]")
    left_axes[1].set_xlabel("batch size")
    right_axes[0].set_ylabel("median total iters")
    right_axes[1].set_ylabel("max total iters")
    right_axes[2].set_ylabel("converged [%]")
    right_axes[2].set_xlabel("batch size")
    right_axes[2].set_ylim(-2, 102)

    for ax in (*left_axes, *right_axes):
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_npz(batch_sizes, results_by_name, out_path):
    save_kw = {
        "batch_sizes": np.asarray(batch_sizes, dtype=np.int32),
        "device": np.array(str(jax.devices()[0])),
        "nx": np.array(NX),
        "ncon": np.array(NCON),
        "neq": np.array(NEQ),
        "solver_tol": np.array(SOLVER_TOL),
        "target_kappa": np.array(TARGET_KAPPA),
        "max_iter": np.array(MAX_ITER),
        "n_warmup": np.array(N_WARMUP),
        "n_repeats": np.array(N_REPEATS),
        "prng_key": np.array(PRNG_KEY),
    }
    for name, results in results_by_name.items():
        for key, value in results.items():
            save_kw[f"{name}_{key}"] = value
    np.savez(out_path, **save_kw)


def main():
    runs = [r for r in RUNS if r.get("enabled", 1)]
    if not runs:
        raise SystemExit("Enable at least one run in RUNS.")

    batch_sizes = np.asarray(BATCH_SIZES, dtype=np.int32)
    results_by_name = {r["name"]: _benchmark_run(r, batch_sizes) for r in runs}

    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    gallery_dir = base / "gallery"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)

    if SAVE_DATA:
        path = data_dir / DATA_FILENAME
        _save_npz(batch_sizes, results_by_name, path)
        print(f"wrote {path}")

    if SAVE_FIGURE:
        path = gallery_dir / FIGURE_FILENAME
        _plot(batch_sizes, results_by_name, runs, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
