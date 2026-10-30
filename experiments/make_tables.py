"""Markdown tables for the README, built from results/summary.csv (written to results/tables.md)."""

from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
s = pd.read_csv(root / "results" / "summary.csv")


def pm(mean, std, nd=3):
    return f"{mean:.{nd}f} ± {0 if pd.isna(std) else std:.{nd}f}"


lines = []
lines.append("### Dice and gap closed by artefact strength and number of unlabeled target images\n")
lines.append("| artefact strength | method | unlabeled target images | Dice | gap closed | RF-FD x1000 | adaptation s |")
lines.append("|---|---|---|---|---|---|---|")
order = {"oracle": 0, "physics-fitted": 1, "GAN translator": 2, "augment": 3, "source-only": 4}
main = s[s.variant.isin(["-", "full", "short"])].copy()
main["o"] = main.method.map(order)
for m, d in main.groupby("misspec"):
    for _, r in d.sort_values(["o", "n_target"]).iterrows():
        gap = "-" if pd.isna(r.gap_closed_mean) or r.method in ("oracle", "source-only") else pm(r.gap_closed_mean, r.gap_closed_std, 2)
        lines.append(f"| {m:g} | {r.method} | {int(r.n_target) if r.n_target else '-'} | {pm(r.dice_mean, r.dice_std)} | {gap} | "
                     f"{pm(r.rf_fd_mean, r.rf_fd_std, 2)} | {r.adapt_seconds_mean:.0f} |")
lines.append("\n### Ablation and GAN budget check (32 unlabeled images, artefact strength 0.5)\n")
lines.append("| method | variant | Dice | gap closed | RF-FD x1000 | adaptation s |")
lines.append("|---|---|---|---|---|---|")
a = s[(s.misspec == 0.5) & (((s.n_target == 32)) | (s.n_target == 0))]
for _, r in a.sort_values(["method", "variant"]).iterrows():
    gap = "-" if pd.isna(r.gap_closed_mean) or r.method in ("oracle", "source-only") else pm(r.gap_closed_mean, r.gap_closed_std, 2)
    lines.append(f"| {r.method} | {r.variant} | {pm(r.dice_mean, r.dice_std)} | {gap} | {pm(r.rf_fd_mean, r.rf_fd_std, 2)} | {r.adapt_seconds_mean:.0f} |")
(root / "results" / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf8")
print("\n".join(lines))
