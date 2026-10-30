"""Figures and summary tables built only from ``results/raw_runs.csv`` (and fitted parameters)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BROWN, TEAL, OCHRE, TERRA, OLIVE, SAND, INK = "#7f5539", "#2a7f7a", "#cc9a2b", "#b5533c", "#6b705c", "#ddb892", "#3b2f2f"
COLORS = {"source-only": OLIVE, "augment": OCHRE, "GAN translator": TERRA, "physics-fitted": TEAL, "oracle": BROWN}
EARTH = LinearSegmentedColormap.from_list("earth", ["#f4ece1", SAND, OCHRE, TEAL, "#1d4e4a"])
DIV = LinearSegmentedColormap.from_list("earthdiv", [TERRA, "#f4ece1", TEAL])
KEYS = ["method", "variant", "n_target", "misspec"]


def add_gap_closed(raw: pd.DataFrame) -> pd.DataFrame:
    """Fraction of the source-only to oracle Dice gap that each run closes (paired by seed and misspec)."""
    raw = raw.copy()
    src = raw[raw.method == "source-only"].set_index(["seed", "misspec"]).dice
    orc = raw[raw.method == "oracle"].set_index(["seed", "misspec"]).dice
    idx = pd.MultiIndex.from_frame(raw[["seed", "misspec"]])
    s, o = src.reindex(idx).to_numpy(), orc.reindex(idx).to_numpy()
    denom = o - s
    raw["gap_closed"] = np.where(denom > 0.05, (raw.dice.to_numpy() - s) / np.where(denom > 0.05, denom, 1), np.nan)
    return raw


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby(KEYS)
    out = g.agg(
        dice_mean=("dice", "mean"), dice_std=("dice", "std"),
        gap_closed_mean=("gap_closed", "mean"), gap_closed_std=("gap_closed", "std"),
        rf_fd_mean=("rf_fd", "mean"), rf_fd_std=("rf_fd", "std"),
        adapt_seconds_mean=("adapt_seconds", "mean"), n_seeds=("seed", "nunique"),
    ).reset_index()
    return out


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def fig_small_multiples(summ: pd.DataFrame, path: Path) -> None:
    ms = sorted(summ.misspec.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, ms):
        d = summ[summ.misspec == m]
        for name in ("source-only", "augment", "oracle"):
            r = d[d.method == name]
            if len(r):
                ax.axhline(r.dice_mean.iloc[0], color=COLORS[name], ls="--", lw=1.4, label=name)
                ax.axhspan(r.dice_mean.iloc[0] - r.dice_std.iloc[0], r.dice_mean.iloc[0] + r.dice_std.iloc[0], color=COLORS[name], alpha=0.12)
        for name, var in (("GAN translator", "short"), ("physics-fitted", "full")):
            r = d[(d.method == name) & (d.variant == var)].sort_values("n_target")
            ax.errorbar(r.n_target, r.dice_mean, r.dice_std, color=COLORS[name], marker="o", capsize=3, lw=2, label=name)
        ax.set_xscale("log", base=2)
        ax.set_title(f"unmodelled-artefact strength {m:g}")
        ax.set_xlabel("unlabeled target images (count)")
        _style(ax)
    axes[0].set_ylabel("Dice on target test set (mean ± std, 0-1)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Physics-fitted simulator vs adversarial translator: line-segmentation quality", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_heatmap(summ: pd.DataFrame, path: Path) -> None:
    ms, ns = sorted(summ.misspec.unique()), sorted(summ[summ.n_target > 0].n_target.unique())
    mats = {}
    for name, var in (("physics-fitted", "full"), ("GAN translator", "short")):
        mat = np.full((len(ms), len(ns)), np.nan)
        for i, m in enumerate(ms):
            for j, n in enumerate(ns):
                r = summ[(summ.method == name) & (summ.variant == var) & (summ.misspec == m) & (summ.n_target == n)]
                if len(r):
                    mat[i, j] = r.gap_closed_mean.iloc[0]
        mats[name] = mat
    diff = mats["physics-fitted"] - mats["GAN translator"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    panels = [("physics-fitted: gap closed", mats["physics-fitted"], EARTH, (0, 1)),
              ("GAN translator: gap closed", mats["GAN translator"], EARTH, (0, 1)),
              ("difference (fitted minus GAN)", diff, DIV, (-1, 1))]
    for ax, (title, mat, cmap, (lo, hi)) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=hi, origin="lower", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=INK, fontsize=10)
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_yticks(range(len(ms)), [f"{m:g}" for m in ms])
        ax.set_xlabel("unlabeled target images (count)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("unmodelled-artefact strength (0-1)")
    fig.suptitle("Fraction of the source-only to oracle Dice gap that each translator closes", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _pareto(points: np.ndarray) -> np.ndarray:
    """Indices of points not dominated (lower cost, higher quality)."""
    keep = []
    for i, (c, q) in enumerate(points):
        if not any((c2 <= c and q2 >= q) and (c2 < c or q2 > q) for j, (c2, q2) in enumerate(points) if j != i):
            keep.append(i)
    return np.array(keep)


def fig_pareto(summ: pd.DataFrame, path: Path, m: float = 0.5) -> None:
    d = summ[(summ.misspec == m) & (summ.method != "oracle")].copy()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name in ("source-only", "augment", "GAN translator", "physics-fitted"):
        r = d[d.method == name]
        ax.errorbar(r.adapt_seconds_mean, r.dice_mean, r.dice_std, fmt="o", color=COLORS[name], alpha=0.85, capsize=2, label=name)
    pts = d[["adapt_seconds_mean", "dice_mean"]].to_numpy()
    front = _pareto(pts)
    fr = pts[front][np.argsort(pts[front][:, 0])]
    ax.step(fr[:, 0], fr[:, 1], where="post", color=INK, lw=1.5, label="Pareto front")
    ax.scatter(fr[:, 0], fr[:, 1], s=130, facecolors="none", edgecolors=INK, lw=1.5)
    orc = summ[(summ.misspec == m) & (summ.method == "oracle")]
    if len(orc):
        ax.axhline(orc.dice_mean.iloc[0], color=BROWN, ls=":", label="oracle (labelled target data)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("adaptation cost: fit or training wall-clock (s, symlog)")
    ax.set_ylabel("Dice on target test set (0-1)")
    ax.set_title(f"Cost versus quality, artefact strength {m:g}", color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_ablation(raw: pd.DataFrame, path: Path, m: float = 0.5, n_t: int = 32) -> None:
    d = raw[(raw.misspec == m)]
    sel = [
        ("source-only", "-", 0), ("augment", "-", 0), ("GAN translator", "short", n_t), ("GAN translator", "long", n_t),
        ("physics-fitted", "full", n_t),
    ] + [("physics-fitted", v, n_t) for v in sorted(d[d.variant.str.startswith("without")].variant.unique())] + [("oracle", "-", 0)]
    labels, means, stds, pts = [], [], [], []
    for method, var, n in sel:
        r = d[(d.method == method) & (d.variant == var) & (d.n_target == n)]
        if len(r):
            labels.append(f"{method}\n{'' if var == '-' else var}".strip())
            means.append(r.dice.mean())
            stds.append(r.dice.std())
            pts.append(r.dice.to_numpy())
    y = np.arange(len(labels))
    cols = [COLORS.get(l.split("\n")[0], TEAL) for l in labels]
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(labels) + 1.4))
    ax.barh(y, means, xerr=stds, color=cols, alpha=0.8, capsize=3)
    for yi, p in zip(y, pts):
        ax.scatter(p, np.full(len(p), yi), color=INK, s=12, zorder=3)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Dice on target test set (0-1); dots are single seeds")
    ax.set_title(f"Ablation: switch off one component of the fitted model ({n_t} target images)", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_gallery(images: dict, path: Path) -> None:
    names = list(images)
    fig, axes = plt.subplots(2, len(names), figsize=(2.3 * len(names), 5.0))
    for j, name in enumerate(names):
        img, mask = images[name]
        for i in range(2):
            ax = axes[i, j]
            ax.imshow(img[i], cmap="cividis", vmin=-1, vmax=1)
            ax.contour(mask[i], levels=[0.5], colors=[TERRA], linewidths=0.6, alpha=0.9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, j].set_title(name, fontsize=9, color=INK)
    axes[0, 0].set_ylabel("sample 1 (slow gate, rows)")
    axes[1, 0].set_ylabel("sample 2")
    fig.suptitle("Simulated charge stability diagrams with exact transition-line masks (contours)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_all(results_dir: Path, figures_dir: Path) -> pd.DataFrame:
    raw = add_gap_closed(pd.read_csv(results_dir / "raw_runs.csv"))
    raw.to_csv(results_dir / "raw_runs.csv", index=False)
    summ = summarise(raw)
    summ.to_csv(results_dir / "summary.csv", index=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_small_multiples(summ[(summ.variant.isin(["-", "full", "short"]))], figures_dir / "fig2_dice_vs_target_images.png")
    fig_heatmap(summ, figures_dir / "fig3_gap_closed_heatmap.png")
    fig_pareto(summ, figures_dir / "fig4_cost_quality_pareto.png", m=0.5 if 0.5 in set(summ.misspec) else float(summ.misspec.iloc[0]))
    n_abl = int(raw[raw.variant.str.startswith("without")].n_target.max()) if raw.variant.str.startswith("without").any() else int(raw.n_target.max())
    fig_ablation(raw, figures_dir / "fig5_ablation.png", m=0.5 if 0.5 in set(raw.misspec) else float(raw.misspec.iloc[0]), n_t=n_abl)
    return summ


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make_all(root / "results", root / "figures")
