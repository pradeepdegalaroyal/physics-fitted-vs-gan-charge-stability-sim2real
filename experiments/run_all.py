"""Reproduce every result: ``python experiments/run_all.py`` (``--quick`` for a smoke run)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))


def _worker(args):
    seed, quick, device = args
    from csdgap.pipeline import FULL, QUICK, run_seed

    return run_seed(seed, QUICK if quick else FULL, device, threads=2 if device == "cuda" else 1, log=lambda s: print(s, flush=True))


def _gallery(cfg, device: str, fits, out: Path) -> None:
    """Example images for the README: source, target proxy, hand augmentation, GAN and fitted simulator."""
    import numpy as np

    from csdgap import render, render_source, render_target
    from csdgap.simulator import PARAM_NAMES
    from csdgap.translators import domain_randomise, train_cyclegan, translate
    from make_figures import fig_gallery

    m, n_t = (0.5, cfg.abl_n_target)
    xs, ms = render_source(8, cfg.size, 5, device)
    xt, mt = render_target(8, m, cfg.size, 6, device)
    pool = render_target(n_t, m, cfg.size, 3000, device)[0]
    g = train_cyclegan(render_source(cfg.n_train, cfg.size, 1000, device)[0], pool, 0, cfg.gan_steps, cfg.gan_batch)
    row = [f for f in fits if f["seed"] == 0 and f["misspec"] == m and f["n_target"] == n_t]
    theta = [row[0][k] for k in PARAM_NAMES] if row else None
    items = {"source (clean)": (xs, ms), "target proxy": (xt, mt), "hand augmentation": (domain_randomise(xs, 0), ms),
             "GAN translator": (translate(g, xs), ms)}
    if theta is not None:
        items["physics-fitted"] = render(theta, 8, cfg.size, 7, device)
    fig_gallery({k: (v[0][:2, 0].cpu().numpy(), v[1][:2, 0].cpu().numpy()) for k, v in items.items()}, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (< 1 minute, CPU is fine)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3, help="parallel seeds (GPU work is launch-bound)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import pandas as pd
    import torch

    from csdgap.pipeline import FULL, QUICK
    from make_figures import make_all

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.2)  # three workers stay below ~5 GB
    cfg = QUICK if args.quick else FULL
    seeds = [0] if args.quick else list(range(args.seeds))
    results = ROOT / "results" / ("quick" if args.quick else "")
    figures = ROOT / "figures" / ("quick" if args.quick else "")
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    jobs = [(s, args.quick, device) for s in seeds]
    if len(jobs) == 1 or args.workers <= 1:
        outs = [_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            outs = list(pool.map(_worker, jobs))
    rows = [r for o in outs for r in o[0]]
    fits = [f for o in outs for f in o[1]]
    pd.DataFrame(rows).to_csv(results / "raw_runs.csv", index=False)
    (results / "fitted_parameters.json").write_text(json.dumps(fits, indent=1))
    print(f"experiments finished in {time.time() - t0:.0f} s", flush=True)

    summ = make_all(results, figures)
    _gallery(cfg, device, fits, figures / "fig1_examples.png")
    meta = dict(device=device, seeds=seeds, quick=args.quick, wall_seconds=round(time.time() - t0, 1),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
    (results / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(summ.to_string(), flush=True)
    print(f"total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
