"""
Plot training metrics from one or more metrics.csv files produced by MAPPOTrainer.

Usage (single run):
    python scripts/plot_metrics.py --csv logs/mappo_seed42/metrics.csv

Usage (multi-seed variance bands):
    python scripts/plot_metrics.py \
        --csv logs/mappo_seed42/metrics.csv \
               logs/mappo_seed123/metrics.csv \
               logs/mappo_seed456/metrics.csv \
        --labels "seed=42" "seed=123" "seed=456" \
        --out-dir plots/

Produces:
    metrics_overview.png  — 2×2 figure with:
        [0,0] mean_waiting_time   vs epoch          (eval rows)
        [0,1] mean_waiting_time   vs wallclock_time (eval rows)
        [1,0] mean_queue_length   vs epoch          (eval rows)
        [1,1] episode_reward      vs epoch          (train rows, smoothed)

Variance shading:
    Single CSV  → ±1 std using std_waiting_time / std_queue_length columns
                  (within-eval-episode variance at each checkpoint)
    Multiple    → ±1 std computed across seeds at each epoch
                  (between-run variance)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe; overridden by --show
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> pd.DataFrame:
    """Load a metrics CSV, coercing numeric columns."""
    df = pd.read_csv(path)
    numeric_cols = [
        "epoch", "wallclock_time_s", "episode_reward", "episode_length",
        "mean_waiting_time", "std_waiting_time",
        "mean_queue_length", "std_queue_length",
        "loss", "actor_loss", "critic_loss", "entropy", "clip_frac",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _shade(ax, x, mean, std, color, alpha: float = 0.18):
    """Draw ±1 std shaded band."""
    if std is not None and not np.all(np.isnan(std)):
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


def _style_ax(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Single-seed aggregation
# ---------------------------------------------------------------------------

def _single_seed_eval(df: pd.DataFrame):
    """Return eval-split data for a single CSV."""
    ev = df[df["split"] == "eval"].copy().sort_values("epoch")
    return ev


def _single_seed_train(df: pd.DataFrame):
    """Return train-split data for a single CSV."""
    tr = df[df["split"] == "train"].copy().sort_values("epoch")
    return tr


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def _multi_seed_eval(dfs: list[pd.DataFrame]):
    """
    Given N dataframes (one per seed), align on epoch and return
    cross-seed mean ± std for each metric.

    Returns a DataFrame with columns:
        epoch, wallclock_time_s (mean),
        mean_waiting_time_mean, mean_waiting_time_std,
        mean_queue_length_mean, mean_queue_length_std,
        episode_reward_mean,    episode_reward_std,
    """
    ev_frames = [_single_seed_eval(df) for df in dfs]

    # Find common epochs
    common_epochs = set(ev_frames[0]["epoch"].dropna())
    for ev in ev_frames[1:]:
        common_epochs &= set(ev["epoch"].dropna())
    common_epochs = sorted(common_epochs)

    rows = []
    for epoch in common_epochs:
        slices = [ev[ev["epoch"] == epoch].iloc[0] for ev in ev_frames
                  if len(ev[ev["epoch"] == epoch]) > 0]
        if len(slices) == 0:
            continue

        def _agg(col):
            vals = np.array([s[col] for s in slices
                             if col in s.index and not pd.isna(s[col])],
                            dtype=float)
            if len(vals) == 0:
                return np.nan, np.nan
            return float(np.mean(vals)), float(np.std(vals))

        wt_mean, wt_std   = _agg("mean_waiting_time")
        ql_mean, ql_std   = _agg("mean_queue_length")
        rew_mean, rew_std = _agg("episode_reward")
        wc_mean, _        = _agg("wallclock_time_s")

        rows.append({
            "epoch":                  epoch,
            "wallclock_time_s":       wc_mean,
            "mean_waiting_time_mean": wt_mean,
            "mean_waiting_time_std":  wt_std,
            "mean_queue_length_mean": ql_mean,
            "mean_queue_length_std":  ql_std,
            "episode_reward_mean":    rew_mean,
            "episode_reward_std":     rew_std,
        })

    return pd.DataFrame(rows)


def _multi_seed_train(dfs: list[pd.DataFrame], smooth: int):
    """
    Aggregate train-split reward across seeds.
    Returns DataFrame with epoch, episode_reward_mean, episode_reward_std.
    """
    tr_frames = []
    for df in dfs:
        tr = _single_seed_train(df)[["epoch", "episode_reward"]].copy()
        tr["episode_reward"] = _rolling_mean(tr["episode_reward"], smooth)
        tr_frames.append(tr)

    all_epochs = sorted(set().union(*[set(tr["epoch"].dropna()) for tr in tr_frames]))
    rows = []
    for epoch in all_epochs:
        vals = [tr.loc[tr["epoch"] == epoch, "episode_reward"].values
                for tr in tr_frames]
        vals = np.concatenate([v for v in vals if len(v) > 0])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        rows.append({
            "epoch": epoch,
            "episode_reward_mean": float(np.mean(vals)),
            "episode_reward_std":  float(np.std(vals)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
]


def plot_single(df: pd.DataFrame, label: str, smooth: int, out_dir: str, show: bool):
    """Plot a single CSV run — no cross-seed variance."""
    ev = _single_seed_eval(df)
    tr = _single_seed_train(df)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Training Metrics — {label}", fontsize=13, fontweight="bold", y=1.01)

    color = PALETTE[0]

    # ── subplot [0,0]: waiting time vs epoch ──────────────────────────────
    ax = axes[0, 0]
    if "mean_waiting_time" in ev.columns and ev["mean_waiting_time"].notna().any():
        x   = ev["epoch"].values
        y   = ev["mean_waiting_time"].values
        std = ev["std_waiting_time"].values if "std_waiting_time" in ev.columns else None
        _shade(ax, x, y, std, color)
        ax.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=3, label=label)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No waiting-time data\n(run eval to populate)",
                ha="center", va="center", transform=ax.transAxes, color="grey")
    _style_ax(ax, "Epoch", "Avg Waiting Time (s)", "Waiting Time vs Epoch")

    # ── subplot [0,1]: waiting time vs wallclock ───────────────────────────
    ax = axes[0, 1]
    if ("mean_waiting_time" in ev.columns and ev["mean_waiting_time"].notna().any()
            and "wallclock_time_s" in ev.columns):
        x   = ev["wallclock_time_s"].values / 60.0   # → minutes
        y   = ev["mean_waiting_time"].values
        std = ev["std_waiting_time"].values if "std_waiting_time" in ev.columns else None
        _shade(ax, x, y, std, color)
        ax.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=3)
    else:
        ax.text(0.5, 0.5, "No waiting-time data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    _style_ax(ax, "Wallclock Time (min)", "Avg Waiting Time (s)",
              "Waiting Time vs Wallclock Time")

    # ── subplot [1,0]: queue length vs epoch ───────────────────────────────
    ax = axes[1, 0]
    if "mean_queue_length" in ev.columns and ev["mean_queue_length"].notna().any():
        x   = ev["epoch"].values
        y   = ev["mean_queue_length"].values
        std = ev["std_queue_length"].values if "std_queue_length" in ev.columns else None
        _shade(ax, x, y, std, color)
        ax.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=3)
    else:
        ax.text(0.5, 0.5, "No queue-length data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    _style_ax(ax, "Epoch", "Avg Queue Length (vehicles)", "Queue Length vs Epoch")

    # ── subplot [1,1]: episode reward vs epoch ─────────────────────────────
    ax = axes[1, 1]
    if "episode_reward" in tr.columns and tr["episode_reward"].notna().any():
        x_raw  = tr["epoch"].values
        y_raw  = tr["episode_reward"].values
        y_smth = _rolling_mean(tr["episode_reward"], smooth).values
        ax.plot(x_raw, y_raw, color=color, linewidth=0.6, alpha=0.3)
        ax.plot(x_raw, y_smth, color=color, linewidth=1.8,
                label=f"rolling mean (w={smooth})")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No reward data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
    _style_ax(ax, "Epoch", "Episode Reward", "Episode Reward vs Epoch")

    fig.tight_layout()
    _save_and_show(fig, out_dir, "metrics_overview.png", show)


def plot_multi(dfs: list[pd.DataFrame], labels: list[str],
               smooth: int, out_dir: str, show: bool):
    """Plot multiple runs with per-seed lines and cross-seed variance shading."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Training Metrics — Multi-Seed Comparison",
                 fontsize=13, fontweight="bold", y=1.01)

    # ── draw individual seed traces (thin, faint) ─────────────────────────
    for i, (df, label) in enumerate(zip(dfs, labels)):
        color = PALETTE[i % len(PALETTE)]
        ev = _single_seed_eval(df)
        tr = _single_seed_train(df)

        if "mean_waiting_time" in ev.columns and ev["mean_waiting_time"].notna().any():
            axes[0, 0].plot(ev["epoch"], ev["mean_waiting_time"],
                            color=color, linewidth=0.7, alpha=0.45, label=label)
            if "wallclock_time_s" in ev.columns:
                axes[0, 1].plot(ev["wallclock_time_s"] / 60.0, ev["mean_waiting_time"],
                                color=color, linewidth=0.7, alpha=0.45)

        if "mean_queue_length" in ev.columns and ev["mean_queue_length"].notna().any():
            axes[1, 0].plot(ev["epoch"], ev["mean_queue_length"],
                            color=color, linewidth=0.7, alpha=0.45)

        if "episode_reward" in tr.columns and tr["episode_reward"].notna().any():
            y_smth = _rolling_mean(tr["episode_reward"], smooth).values
            axes[1, 1].plot(tr["epoch"], y_smth,
                            color=color, linewidth=0.7, alpha=0.45)

    # ── draw cross-seed aggregate (bold mean + shaded std) ────────────────
    agg_ev = _multi_seed_eval(dfs)
    agg_tr = _multi_seed_train(dfs, smooth)
    agg_color = "#212121"  # near-black for the aggregate

    def _plot_agg_eval(ax, mean_col, std_col, xlabel, ylabel, title):
        if mean_col in agg_ev.columns and agg_ev[mean_col].notna().any():
            x   = agg_ev["epoch"].values
            y   = agg_ev[mean_col].values
            std = agg_ev[std_col].values if std_col in agg_ev.columns else None
            _shade(ax, x, y, std, agg_color, alpha=0.15)
            ax.plot(x, y, color=agg_color, linewidth=2.2,
                    marker="o", markersize=3.5, label="mean ± std (across seeds)",
                    zorder=5)
            ax.legend(fontsize=8)
        _style_ax(ax, xlabel, ylabel, title)

    _plot_agg_eval(axes[0, 0],
                   "mean_waiting_time_mean", "mean_waiting_time_std",
                   "Epoch", "Avg Waiting Time (s)", "Waiting Time vs Epoch")

    # wallclock variant — x-axis is wallclock of the aggregate mean
    if "mean_waiting_time_mean" in agg_ev.columns and agg_ev["mean_waiting_time_mean"].notna().any():
        x   = agg_ev["wallclock_time_s"].values / 60.0
        y   = agg_ev["mean_waiting_time_mean"].values
        std = agg_ev["mean_waiting_time_std"].values
        _shade(axes[0, 1], x, y, std, agg_color, alpha=0.15)
        axes[0, 1].plot(x, y, color=agg_color, linewidth=2.2,
                        marker="o", markersize=3.5, zorder=5)
    _style_ax(axes[0, 1], "Wallclock Time (min)", "Avg Waiting Time (s)",
              "Waiting Time vs Wallclock Time")

    _plot_agg_eval(axes[1, 0],
                   "mean_queue_length_mean", "mean_queue_length_std",
                   "Epoch", "Avg Queue Length (vehicles)", "Queue Length vs Epoch")

    if "episode_reward_mean" in agg_tr.columns and agg_tr["episode_reward_mean"].notna().any():
        x   = agg_tr["epoch"].values
        y   = agg_tr["episode_reward_mean"].values
        std = agg_tr["episode_reward_std"].values
        _shade(axes[1, 1], x, y, std, agg_color, alpha=0.15)
        axes[1, 1].plot(x, y, color=agg_color, linewidth=2.2,
                        label=f"mean ± std (rolling w={smooth})", zorder=5)
        axes[1, 1].legend(fontsize=8)
    _style_ax(axes[1, 1], "Epoch", "Episode Reward", "Episode Reward vs Epoch")

    # Add per-seed legend to subplot [0,0]
    axes[0, 0].legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    _save_and_show(fig, out_dir, "metrics_overview.png", show)


# ---------------------------------------------------------------------------
# Save / show
# ---------------------------------------------------------------------------

def _save_and_show(fig: plt.Figure, out_dir: str, filename: str, show: bool):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot_metrics] Saved → {out_path}")
    if show:
        matplotlib.use("TkAgg")  # switch to interactive backend
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot training metrics from MAPPOTrainer CSV logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Single run:\n"
            "    python scripts/plot_metrics.py --csv logs/mappo_seed42/metrics.csv\n\n"
            "  Multi-seed (explicit):\n"
            "    python scripts/plot_metrics.py \\\n"
            "        --csv logs/baseline_seed0/metrics.csv \\\n"
            "               logs/baseline_seed1/metrics.csv\n\n"
            "  Auto-discover all metrics.csv files under a directory (e.g. after sweep):\n"
            "    python scripts/plot_metrics.py --log-dir logs/\n"
        ),
    )
    # --- input sources (mutually exclusive) ---
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--csv", nargs="+", metavar="FILE",
        help="Explicit path(s) to metrics.csv file(s).",
    )
    src.add_argument(
        "--log-dir", metavar="DIR",
        help=(
            "Root directory to search recursively for metrics.csv files. "
            "Each file found becomes one run in the plot. "
            "Labels default to the immediate parent directory name."
        ),
    )
    p.add_argument(
        "--labels", nargs="*", default=None, metavar="LABEL",
        help="Display labels for each CSV (must match input count). "
             "Defaults to parent directory names.",
    )
    p.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Output directory for the saved PNG. "
             "Defaults to --log-dir (when used) or directory of first CSV.",
    )
    p.add_argument(
        "--smooth", type=int, default=10, metavar="N",
        help="Rolling-mean window for the reward subplot (default: 10).",
    )
    p.add_argument(
        "--show", action="store_true",
        help="Display the figure interactively (requires a display).",
    )
    return p.parse_args()


def _discover_csvs(log_dir: str):
    """
    Recursively find all metrics.csv files under log_dir.
    Returns sorted list of (csv_path, label) tuples where label is the
    immediate parent directory name (e.g. 'baseline_seed0').
    """
    root = Path(log_dir)
    found = sorted(root.rglob("metrics.csv"))
    if not found:
        print(f"[ERROR] No metrics.csv files found under {log_dir}", file=sys.stderr)
        sys.exit(1)
    return [(str(p), p.parent.name) for p in found]


def main():
    args = parse_args()

    # --- Resolve CSV paths and labels ---
    if args.log_dir:
        pairs = _discover_csvs(args.log_dir)
        csv_paths = [p for p, _ in pairs]
        auto_labels = [lbl for _, lbl in pairs]
        print(f"[plot_metrics] Discovered {len(csv_paths)} CSV(s) under {args.log_dir}:")
        for p, lbl in pairs:
            print(f"  {lbl}: {p}")
        out_dir = args.out_dir or args.log_dir
    else:
        csv_paths = args.csv
        auto_labels = [Path(p).parent.name or Path(p).name for p in csv_paths]
        out_dir = args.out_dir or str(Path(csv_paths[0]).parent)

    # Validate paths exist
    for path in csv_paths:
        if not os.path.isfile(path):
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Resolve labels
    labels = args.labels if args.labels is not None else auto_labels
    if len(labels) != len(csv_paths):
        print(
            f"[ERROR] --labels count ({len(labels)}) must match "
            f"CSV count ({len(csv_paths)}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load
    dfs = [_load_csv(p) for p in csv_paths]
    print(f"[plot_metrics] Loaded {len(dfs)} CSV(s)")

    if len(dfs) == 1:
        plot_single(dfs[0], labels[0], args.smooth, out_dir, args.show)
    else:
        plot_multi(dfs, labels, args.smooth, out_dir, args.show)

    print("[plot_metrics] Done.")


if __name__ == "__main__":
    main()
