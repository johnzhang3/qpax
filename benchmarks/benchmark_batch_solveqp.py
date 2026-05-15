"""Benchmark full batched QP solves on the active JAX device.

Measures the whole solver loop, not just individual kernels. It can benchmark
either:
  * `solve_qp` only
  * `solve_qp` followed by `relax_qp`

Reports per-stage wall-clock time, convergence flags, and iteration counts.

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

from qpax.explicit import pdip as e_pdip
from qpax.explicit import pdip_relaxed as e_pdip_relaxed
from qpax.implicit import pdip as i_pdip
from qpax.implicit import pdip_relaxed as i_pdip_relaxed

# ================================ Config ================================== #
NX = 32
NCON = 64
NEQ = 0

BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

N_WARMUP = 1
N_REPEATS = 1

MAX_ITER = 100
SOLVER_TOL = 1e-5
TARGET_KAPPA = 1e-3

# "solve_qp" or "solve_then_relax".
BENCHMARK_MODE = "solve_then_relax"

PRNG_KEY = 7
PROBLEM_KIND = "projection_like"  # "projection_like" or "generic_random"

INIT_POINT = 0.5
G_MIN = -4.0
G_MAX = 4.0
SLACK_MIN = 0.5
SLACK_MAX = 1.5

GENERIC_G_SCALE = 0.25
GENERIC_H_SCALE = 1.0

verbose = len(BATCH_SIZES) <= 2

RUNS = (
    {"name": "e64", "backend": "e", "use_f64": True, "enabled": 1, "verbose": verbose},
    {"name": "e32", "backend": "e", "use_f64": False, "enabled": 1, "verbose": verbose},
    {"name": "i64", "backend": "i", "use_f64": True, "enabled": 1, "verbose": verbose},
    {"name": "i32", "backend": "i", "use_f64": False, "enabled": 1, "verbose": verbose},
)

PRINT_FULL_LISTS = False

SAVE_DATA = True
SAVE_FIGURE = True
if BENCHMARK_MODE == "solve_qp":
    DATA_FILENAME = "benchmark_batch_solveqp.npz"
    FIGURE_FILENAME = "benchmark_batch_solveqp.png"
elif BENCHMARK_MODE == "solve_then_relax":
    DATA_FILENAME = "benchmark_batch_solveqp_relaxqp.npz"
    FIGURE_FILENAME = "benchmark_batch_solveqp_relaxqp.png"
else:
    raise ValueError(f"Unsupported BENCHMARK_MODE {BENCHMARK_MODE!r}.")


BACKENDS = {
    "e": (e_pdip, e_pdip_relaxed),
    "i": (i_pdip, i_pdip_relaxed),
}


def _dtype(use_f64: bool):
    jax.config.update("jax_enable_x64", bool(use_f64))
    return jnp.float64 if use_f64 else jnp.float32


def _stage_names():
    if BENCHMARK_MODE == "solve_qp":
        return ("solve_qp",)
    return ("solve_qp", "relax_qp")


def _make_problem_bank(max_batch: int, dtype):
    key = jax.random.PRNGKey(PRNG_KEY)
    keys = iter(jax.random.split(key, 16))

    def normal(shape, scale=1.0):
        return scale * jax.random.normal(next(keys), shape, dtype=dtype)

    def uniform(shape, minval=0.0, maxval=1.0):
        return jax.random.uniform(
            next(keys), shape, minval=minval, maxval=maxval, dtype=dtype
        )

    Q = jnp.broadcast_to(jnp.eye(NX, dtype=dtype), (max_batch, NX, NX))
    A = jnp.zeros((max_batch, NEQ, NX), dtype=dtype)
    b = jnp.zeros((max_batch, NEQ), dtype=dtype)

    if PROBLEM_KIND == "projection_like":
        q = -normal((max_batch, NX))
        z0 = jnp.full((max_batch, NX), INIT_POINT, dtype=dtype)
        G = uniform((max_batch, NCON, NX), minval=G_MIN, maxval=G_MAX)
        s0 = uniform((max_batch, NCON), minval=SLACK_MIN, maxval=SLACK_MAX)
        h = jnp.einsum("bij,bj->bi", G, z0) + s0
    elif PROBLEM_KIND == "generic_random":
        q = normal((max_batch, NX))
        G = normal((max_batch, NCON, NX), scale=GENERIC_G_SCALE)
        h = normal((max_batch, NCON), scale=GENERIC_H_SCALE)
    else:
        raise ValueError(f"Unsupported PROBLEM_KIND {PROBLEM_KIND!r}.")

    return {"Q": Q, "q": q, "A": A, "b": b, "G": G, "h": h}


def _slice_problem(bank, batch_size: int):
    return {k: v[:batch_size] for k, v in bank.items()}


def _build_batched_solver(backend: str):
    pdip_mod, _ = BACKENDS[backend]

    def solve_one(Q, q, A, b, G, h):
        return pdip_mod.solve_qp(
            Q, q, A, b, G, h, solver_tol=SOLVER_TOL, max_iter=MAX_ITER
        )

    return jax.jit(jax.vmap(solve_one))


def _build_batched_relaxer(backend: str):
    _, relaxed_mod = BACKENDS[backend]

    def relax_one(Q, q, A, b, G, h, x, s, z, y):
        return relaxed_mod.relax_qp(
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

    return jax.jit(jax.vmap(relax_one))


def _run_debug_batch(backend: str, batch, batch_size: int):
    pdip_mod, relaxed_mod = BACKENDS[backend]
    do_relax = BENCHMARK_MODE == "solve_then_relax"

    print(f"    debug pass: running debug solvers for {batch_size} QPs")
    for i in range(batch_size):
        print(f"    [qp {i + 1}/{batch_size}]")
        out = pdip_mod.solve_qp(
            batch["Q"][i],
            batch["q"][i],
            batch["A"][i],
            batch["b"][i],
            batch["G"][i],
            batch["h"][i],
            solver_tol=SOLVER_TOL,
            max_iter=MAX_ITER,
            verbose=True,
        )
        if do_relax:
            relaxed_mod.relax_qp(
                batch["Q"][i],
                batch["q"][i],
                batch["A"][i],
                batch["b"][i],
                batch["G"][i],
                batch["h"][i],
                *out[:4],
                solver_tol=SOLVER_TOL,
                target_kappa=TARGET_KAPPA,
                max_iter=MAX_ITER,
                verbose=True,
            )


def _time_call(fn, *args):
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return 1000.0 * (time.perf_counter() - t0), out


def _format_array(values, batch_size: int):
    arr = np.asarray(values)[:batch_size]
    if PRINT_FULL_LISTS or batch_size <= 16:
        return np.array2string(arr, separator=", ")
    head = np.array2string(arr[:8], separator=", ")
    tail = np.array2string(arr[-4:], separator=", ")
    return f"{head[:-1]}, ..., {tail[1:]}"


def _empty_stage_results(n_batch_sizes: int, max_batch: int):
    return {
        "wall_ms": np.zeros(n_batch_sizes, dtype=np.float64),
        "wall_samples_ms": np.zeros((n_batch_sizes, N_REPEATS), dtype=np.float64),
        "iters_pad": np.full((n_batch_sizes, max_batch), -1, dtype=np.int32),
        "conv_pad": np.full((n_batch_sizes, max_batch), -1, dtype=np.int32),
        "max_iters": np.zeros(n_batch_sizes, dtype=np.int32),
        "median_iters": np.zeros(n_batch_sizes, dtype=np.float64),
        "converged_frac": np.zeros(n_batch_sizes, dtype=np.float64),
    }


def _store_stage_outputs(stage_results, batch_idx, batch_size, out):
    converged, iters = out[-2:]
    converged_np = np.asarray(converged, dtype=np.int32)
    iters_np = np.asarray(iters, dtype=np.int32)

    stage_results["wall_ms"][batch_idx] = float(
        np.median(stage_results["wall_samples_ms"][batch_idx])
    )
    stage_results["iters_pad"][batch_idx, :batch_size] = iters_np
    stage_results["conv_pad"][batch_idx, :batch_size] = converged_np
    stage_results["max_iters"][batch_idx] = int(np.max(iters_np))
    stage_results["median_iters"][batch_idx] = float(np.median(iters_np))
    stage_results["converged_frac"][batch_idx] = float(np.mean(converged_np))


def _stage_summary_str(stage_name, stage_results, batch_idx):
    return (
        f"{stage_name}: wall_ms={stage_results['wall_ms'][batch_idx]:8.3f} "
        f"median_iter={stage_results['median_iters'][batch_idx]:6.1f} "
        f"max_iter={stage_results['max_iters'][batch_idx]:4d} "
        f"conv={100.0 * stage_results['converged_frac'][batch_idx]:6.1f}%"
    )


def _print_debug_arrays(stage_name, stage_results, batch_idx, batch_size):
    conv_row = stage_results["conv_pad"][batch_idx, :batch_size]
    iters_row = stage_results["iters_pad"][batch_idx, :batch_size]
    print(f"    {stage_name} converged={_format_array(conv_row, batch_size)}")
    print(f"    {stage_name} iters    ={_format_array(iters_row, batch_size)}")


def _benchmark_solver(run_cfg, batch_sizes):
    dtype = _dtype(bool(run_cfg["use_f64"]))
    backend = run_cfg["backend"]
    solver_fn = _build_batched_solver(backend)
    relax_fn = (
        _build_batched_relaxer(backend)
        if BENCHMARK_MODE == "solve_then_relax"
        else None
    )

    max_batch = int(np.max(batch_sizes))
    problem_bank = _make_problem_bank(max_batch, dtype)
    results_by_stage = {
        stage: _empty_stage_results(len(batch_sizes), max_batch)
        for stage in _stage_names()
    }

    print(
        f"=== solving {run_cfg['name']} "
        f"(device={jax.devices()[0]}, dtype={dtype}, "
        f"problem_kind={PROBLEM_KIND}, mode={BENCHMARK_MODE}) ==="
    )

    for i, batch_size in enumerate(batch_sizes):
        bs = int(batch_size)
        batch = _slice_problem(problem_bank, bs)
        args = (batch["Q"], batch["q"], batch["A"], batch["b"], batch["G"], batch["h"])

        if run_cfg.get("verbose", False):
            _run_debug_batch(backend, batch, bs)

        solve_out = relax_out = None
        for _ in range(N_WARMUP):
            solve_out = solver_fn(*args)
            jax.block_until_ready(solve_out)
            if relax_fn is not None:
                relax_out = relax_fn(*args, *solve_out[:4])
                jax.block_until_ready(relax_out)

        for j in range(N_REPEATS):
            sample_ms, solve_out = _time_call(solver_fn, *args)
            results_by_stage["solve_qp"]["wall_samples_ms"][i, j] = sample_ms
            if relax_fn is not None:
                sample_ms, relax_out = _time_call(relax_fn, *args, *solve_out[:4])
                results_by_stage["relax_qp"]["wall_samples_ms"][i, j] = sample_ms

        _store_stage_outputs(results_by_stage["solve_qp"], i, bs, solve_out)
        if relax_fn is not None:
            _store_stage_outputs(results_by_stage["relax_qp"], i, bs, relax_out)

        stage_parts = " | ".join(
            _stage_summary_str(stage, results_by_stage[stage], i)
            for stage in _stage_names()
        )
        print(f"  batch={bs:4d} {stage_parts}")

        if run_cfg.get("verbose", False):
            for stage in _stage_names():
                _print_debug_arrays(stage, results_by_stage[stage], i, bs)

    return results_by_stage


def _plot_style(run_cfg):
    color = "#1f77b4" if run_cfg["backend"] == "e" else "#ff7f0e"
    linestyle = "-" if run_cfg["use_f64"] else "-."
    return color, linestyle, 2.0


def _plot(batch_sizes, results_by_name, runs, out_path):
    stage_names = _stage_names()
    fig, axes = plt.subplots(
        4,
        len(stage_names),
        figsize=(7 * len(stage_names), 14),
        sharex="col",
        squeeze=False,
    )
    ylabels = (
        "wall time [ms]",
        "median PDIP iterations",
        "max PDIP iterations",
        "converged QPs [%]",
    )
    series = ("wall_ms", "median_iters", "max_iters", "converged_frac")

    for col, stage in enumerate(stage_names):
        axes[0, col].set_title(stage)
        for run_cfg in runs:
            name = run_cfg["name"]
            if name not in results_by_name:
                continue
            stage_results = results_by_name[name][stage]
            color, linestyle, linewidth = _plot_style(run_cfg)
            for row, key in enumerate(series):
                values = stage_results[key]
                if key == "converged_frac":
                    values = 100.0 * values
                axes[row, col].plot(
                    batch_sizes,
                    values,
                    marker="o",
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=name,
                )

        for row, ylabel in enumerate(ylabels):
            ax = axes[row, col]
            ax.set_xscale("log", base=2)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[3, col].set_xlabel("batch size")
        axes[3, col].set_ylim(-2, 102)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_npz(batch_sizes, results_by_name, out_path):
    save_kw = {
        "batch_sizes": np.asarray(batch_sizes, dtype=np.int32),
        "device": np.array(str(jax.devices()[0])),
        "benchmark_mode": np.array(BENCHMARK_MODE),
        "problem_kind": np.array(PROBLEM_KIND),
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
    for name, stage_results_by_name in results_by_name.items():
        for stage, results in stage_results_by_name.items():
            for key, value in results.items():
                save_kw[f"{name}_{stage}_{key}"] = value
    np.savez(out_path, **save_kw)


def main():
    runs = [r for r in RUNS if r["enabled"]]
    if not runs:
        raise SystemExit("Enable at least one run in RUNS.")

    batch_sizes = np.asarray(BATCH_SIZES, dtype=np.int32)
    if np.any(batch_sizes <= 0):
        raise SystemExit("BATCH_SIZES must be positive integers.")

    results_by_name = {r["name"]: _benchmark_solver(r, batch_sizes) for r in runs}

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
