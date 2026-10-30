"""One seed of the study: train segmenters on every kind of training data, test on the target proxy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .features import RandomFeatureFD
from .fit import fit_simulator, param_table
from .segmenter import dice_score, train_segmenter
from .simulator import COMPONENTS, render, render_source, render_target
from .translators import domain_randomise, train_cyclegan, translate


@dataclass(frozen=True)
class Config:
    """Experiment sizes.  ``full`` is the published run; ``quick`` is a smoke test."""

    size: int = 64
    n_train: int = 768
    n_test: int = 512
    seg_steps: int = 150
    seg_batch: int = 96
    gan_steps: int = 180
    gan_batch: int = 24
    gan_long_steps: int = 800
    fit_iters: int = 14
    fit_pop: int = 6
    fit_sim: int = 24
    n_targets: Tuple[int, ...] = (8, 32, 128)
    misspec: Tuple[float, ...] = (0.0, 0.5, 1.0)
    ablation: Tuple[str, ...] = ("thermal broadening", "sensor peak", "charge jumps", "physics shift")
    abl_misspec: float = 0.5
    abl_n_target: int = 32
    gan_long: bool = True


QUICK = Config(
    size=32, n_train=64, n_test=64, seg_steps=10, seg_batch=16, gan_steps=4, gan_batch=8,
    gan_long_steps=4, fit_iters=1, fit_pop=3, fit_sim=4, n_targets=(8,), misspec=(0.5,),
    ablation=("charge jumps",), abl_n_target=8, gan_long=False,
)
FULL = Config()


def _row(method: str, variant: str, n_target: int, misspec: float, seed: int, dice: float,
         fd: float, adapt_s: float, extra: float = float("nan")) -> Dict:
    return dict(method=method, variant=variant, n_target=n_target, misspec=misspec, seed=seed,
                dice=dice, rf_fd=fd, adapt_seconds=adapt_s, fit_objective=extra)


def run_seed(seed: int, cfg: Config = FULL, device: str = "cuda", threads: int = 2, log=print) -> Tuple[List[Dict], List[Dict]]:
    """Run every method for one seed.

    Returns:
        ``(rows, fits)``: one row per trained segmenter, and the fitted simulator parameters.
    """
    torch.set_num_threads(threads)
    rows: List[Dict] = []
    fits: List[Dict] = []
    fd_metric = RandomFeatureFD(device)
    s = cfg.size

    def seg(images, masks, tag):
        return train_segmenter(images, masks, seed=seed, steps=cfg.seg_steps, batch=cfg.seg_batch)

    xs, ms = render_source(cfg.n_train, s, 1000 + seed, device)
    tests = {m: render_target(cfg.n_test, m, s, 999, device) for m in cfg.misspec}

    # baselines that never see target data
    for name, imgs in (("source-only", xs), ("augment", domain_randomise(xs, seed))):
        net = seg(imgs, ms, name)
        for m, (xt, mt) in tests.items():
            rows.append(_row(name, "-", 0, m, seed, dice_score(net, xt, mt), fd_metric(imgs[: cfg.n_test], xt), 0.0))
    log(f"seed {seed}: baselines done")

    def evaluate(method, variant, n_t, m, imgs, masks, adapt_s, extra=float("nan")):
        net = seg(imgs, masks, method)
        xt, mt = tests[m]
        rows.append(_row(method, variant, n_t, m, seed, dice_score(net, xt, mt), fd_metric(imgs[: cfg.n_test], xt), adapt_s, extra))

    pools = {}
    for m in cfg.misspec:
        xo, mo = render_target(cfg.n_train, m, s, 2000 + seed, device)
        evaluate("oracle", "-", 0, m, xo, mo, 0.0)
        pools[m] = render_target(max(cfg.n_targets), m, s, 3000 + seed, device)[0]
        for n_t in cfg.n_targets:
            tgt = pools[m][:n_t]
            t0 = time.time()
            theta, obj = fit_simulator(tgt, seed, (), cfg.fit_sim, cfg.fit_iters, cfg.fit_pop, s, device)
            dt = time.time() - t0
            xf, mf = render(theta, cfg.n_train, s, 4000 + seed, device)
            evaluate("physics-fitted", "full", n_t, m, xf, mf, dt, obj)
            fits.append(dict(seed=seed, misspec=m, n_target=n_t, **param_table(theta)))

            t0 = time.time()
            g = train_cyclegan(xs, tgt, seed, cfg.gan_steps, cfg.gan_batch)
            dt = time.time() - t0
            evaluate("GAN translator", "short", n_t, m, translate(g, xs), ms, dt)
        log(f"seed {seed}: misspec {m} done")

    m, n_t = cfg.abl_misspec, cfg.abl_n_target
    if m in pools:
        tgt = pools[m][:n_t]
        for comp in cfg.ablation:
            t0 = time.time()
            theta, obj = fit_simulator(tgt, seed, (comp,), cfg.fit_sim, cfg.fit_iters, cfg.fit_pop, s, device)
            dt = time.time() - t0
            xf, mf = render(theta, cfg.n_train, s, 4000 + seed, device)
            evaluate("physics-fitted", f"without {comp}", n_t, m, xf, mf, dt, obj)
        if cfg.gan_long:
            t0 = time.time()
            g = train_cyclegan(xs, tgt, seed, cfg.gan_long_steps, cfg.gan_batch)
            dt = time.time() - t0
            evaluate("GAN translator", "long", n_t, m, translate(g, xs), ms, dt)
    log(f"seed {seed}: finished")
    return rows, fits
