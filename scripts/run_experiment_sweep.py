#!/usr/bin/env python3
"""
Parallel experiment sweep launcher for the MAPPO / MAPPO_BNTT A/B comparison.

Spawns one subprocess per (variant × seed) combination.  Each subprocess is a
call to scripts/train_mappo.py with the appropriate flags.  A semaphore keeps
at most --max-parallel processes running simultaneously.

Output
------
Each run writes its stdout + stderr to:
    logs/<variant>_seed<N>/run.log

A summary table is printed when all subprocesses finish.

Usage
-----
    # Default: 2 variants × 3 seeds (0,1,2), max 4 parallel
    python scripts/run_experiment_sweep.py

    # Custom seeds and parallelism
    python scripts/run_experiment_sweep.py --seeds 0 1 2 3 4 --max-parallel 3

    # Only the BN-critic variant
    python scripts/run_experiment_sweep.py --variants critic-bn --seeds 0 1 2

    # Quick smoke-test (1 epoch per run)
    python scripts/run_experiment_sweep.py --max-epoch 1 --seeds 0

Variants
--------
  baseline   — standard MLP critic (train_mappo.py, no extra flags)
  critic-bn  — BatchNorm1d critic (train_mappo.py --use-critic-bn)
"""

import argparse
import datetime
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List

# Resolve repo root so the script works when called from any directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_mappo.py"

VARIANT_FLAGS = {
    "baseline": [],
    "critic-bn": ["--use-critic-bn"],
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MAPPO baseline vs critic-BN experiment in parallel."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANT_FLAGS.keys()),
        default=list(VARIANT_FLAGS.keys()),
        help="Which variants to run (default: all).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds (default: 0 1 2).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help=(
            "Max simultaneous processes.  "
            "Defaults to min(#runs, os.cpu_count() // 2, 4)."
        ),
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Override max_epoch for every run (useful for smoke-tests).",
    )
    parser.add_argument(
        "--log-base-dir",
        type=str,
        default="logs",
        help="Base directory under which per-run log dirs are created.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device override forwarded to every subprocess.",
    )
    parser.add_argument(
        "--fast-test",
        action="store_true",
        default=False,
        help=(
            "Forward --fast-test to every subprocess, activating "
            "get_fast_test_config() (10 epochs, test_interval=1, n_test_envs=5). "
            "Recommended for the BN vs baseline comparison."
        ),
    )
    # W&B is off by default.  Pass --wandb to enable it for all runs.
    # --no-wandb is an explicit opt-out (useful when the config has use_wandb=True).
    # Implemented as a mutually exclusive group so both flags can't be set together.
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable W&B logging for all runs.",
    )
    wandb_group.add_argument(
        "--no-wandb",
        action="store_true",
        default=False,
        help="Explicitly disable W&B for all runs (default behaviour).",
    )
    return parser.parse_args()


# ── Per-run subprocess ─────────────────────────────────────────────────────────

def _stream_output(proc: subprocess.Popen, log_path: Path, prefix: str) -> None:
    """
    Read proc stdout line-by-line, write to log file, and echo to terminal.

    Runs in a dedicated thread per subprocess so output from parallel runs is
    interleaved in the terminal (each line is prefixed with the run tag).
    """
    with log_path.open("w") as log_fh:
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace")
            log_fh.write(line)
            log_fh.flush()
            # Print to terminal with run prefix, stripping trailing newline
            # so our prefix + the line forms a clean single line.
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()


def launch_run(
    variant: str,
    seed: int,
    log_base_dir: str,
    extra_flags: List[str],
    semaphore: threading.Semaphore,
    results: dict,
) -> threading.Thread:
    """
    Return a Thread that acquires the semaphore, launches the subprocess,
    streams its output, and records the exit code in `results`.
    """
    run_name = f"{variant}_seed{seed}"
    # Resolve the BASE log directory (not the run subdir).
    # trainer.py computes:  log_dir = os.path.join(config.logging.log_dir, exp_name)
    # so passing the base here means the metrics.csv ends up at:
    #   <log_base_dir>/<run_name>/metrics.csv   (single level, no double-nesting)
    base_dir = Path(log_base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    # The run.log lives next to where the metrics.csv will appear.
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--seed", str(seed),
        "--experiment-name", run_name,
        "--log-dir", str(base_dir),   # base only — trainer appends run_name
    ] + VARIANT_FLAGS[variant] + extra_flags

    def _run():
        with semaphore:
            print(f"\n[sweep] ▶  Starting  {run_name}")
            print(f"[sweep]    cmd: {' '.join(cmd)}")
            print(f"[sweep]    log: {log_path}")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # merge stderr → stdout
                cwd=str(REPO_ROOT),
            )

            _stream_output(proc, log_path, prefix=run_name)
            proc.wait()

            results[run_name] = {
                "variant": variant,
                "seed": seed,
                "exit_code": proc.returncode,
                "log": str(log_path),
            }
            status = "✓" if proc.returncode == 0 else "✗"
            print(f"\n[sweep] {status}  Finished {run_name}  (exit {proc.returncode})")

    t = threading.Thread(target=_run, name=run_name, daemon=False)
    return t


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Build the list of (variant, seed) pairs to run.
    runs = [(v, s) for v in args.variants for s in args.seeds]
    n_runs = len(runs)

    # Create a timestamped subdirectory for this sweep so that
    # `plot_metrics.py --log-dir <sweep_dir>` never picks up stale runs
    # from previous sweeps or standalone training sessions.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = str(Path(args.log_base_dir) / f"sweep_{timestamp}")

    # Determine parallelism.
    # Use 'is not None' (not 'or') so an explicit --max-parallel 0 isn't
    # silently ignored (0 is falsy, but a valid — if unusual — value meaning
    # "run everything sequentially via the semaphore").
    cpu_default = max(1, (os.cpu_count() or 2) // 2)
    max_parallel = args.max_parallel if args.max_parallel is not None else min(n_runs, cpu_default, 4)

    # Build extra CLI flags forwarded to every subprocess.
    extra_flags: List[str] = []
    if args.fast_test:
        extra_flags.append("--fast-test")
    if args.max_epoch is not None:
        extra_flags += ["--max-epoch", str(args.max_epoch)]
    if args.device is not None:
        extra_flags += ["--device", args.device]
    if args.wandb:
        extra_flags.append("--wandb")
    elif args.no_wandb:
        extra_flags.append("--no-wandb")

    print("\n" + "=" * 70)
    print("MAPPO Experiment Sweep")
    print("=" * 70)
    print(f"  Variants:     {args.variants}")
    print(f"  Seeds:        {args.seeds}")
    print(f"  Total runs:   {n_runs}")
    print(f"  Max parallel: {max_parallel}")
    print(f"  Sweep dir:    {sweep_dir}")
    if extra_flags:
        print(f"  Extra flags:  {extra_flags}")
    print("=" * 70 + "\n")

    semaphore = threading.Semaphore(max_parallel)
    results: dict = {}

    # Launch all threads; the semaphore gates actual subprocess creation.
    threads = []
    for variant, seed in runs:
        t = launch_run(
            variant=variant,
            seed=seed,
            log_base_dir=sweep_dir,   # timestamped subdir — isolated from old runs
            extra_flags=extra_flags,
            semaphore=semaphore,
            results=results,
        )
        threads.append(t)
        t.start()
        # Small stagger to avoid simultaneous TraCI port collisions at startup.
        time.sleep(0.5)

    # Wait for all runs to complete.
    for t in threads:
        t.join()

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Sweep Summary")
    print("=" * 70)
    header = f"{'Run':<30} {'Exit':>5}  {'Log'}"
    print(header)
    print("-" * 70)

    n_ok = n_fail = 0
    for run_name in sorted(results):
        r = results[run_name]
        ok = r["exit_code"] == 0
        icon = "✓" if ok else "✗"
        print(f"{icon} {run_name:<28} {r['exit_code']:>5}  {r['log']}")
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print("=" * 70)
    print(f"Completed: {n_ok}/{n_runs} succeeded, {n_fail} failed.")

    if n_fail > 0:
        print("\nFailed runs:")
        for run_name, r in results.items():
            if r["exit_code"] != 0:
                print(f"  {run_name}  →  {r['log']}")
        sys.exit(1)

    print(
        "\nTo compare results, run:\n"
        f"  python scripts/plot_metrics.py --log-dir {sweep_dir}"
        "\n\nOr for a specific subset of runs:\n"
        "  python scripts/plot_metrics.py \\\n"
        + "".join(
            # Single nesting: <sweep_dir>/<run_name>/metrics.csv
            f"    --csv {sweep_dir}/{v}_seed{s}/metrics.csv \\\n"
            for v in args.variants for s in args.seeds
        )
        + f"  --out-dir {sweep_dir}/plots/"
    )


if __name__ == "__main__":
    main()
