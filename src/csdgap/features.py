"""Image statistics used to fit the simulator and to score realism.

``summary_features`` is the statistic the physics-structured fit matches.
``RandomFeatureFD`` is an *independent* realism score (Frechet distance in the
feature space of a fixed random convolutional network), so the fitted model is
not graded on the quantity it optimised.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg

N_FEATURES = 24


_BIN_CACHE: dict = {}


def _radial_bins(h: int, w: int, device) -> torch.Tensor:
    """Dense (h*w, 6) averaging matrix for radial power-spectrum bins (cached)."""
    key = (h, w, str(device))
    if key not in _BIN_CACHE:
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        rad = ((yy - h // 2) ** 2 + (xx - w // 2) ** 2).float().sqrt().flatten()
        edges = [0.5, 2, 4, 7, 11, 16, 23]
        cols = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = ((rad >= lo * h / 64) & (rad < hi * h / 64)).float()
            cols.append(sel / sel.sum().clamp(min=1))
        _BIN_CACHE[key] = torch.stack(cols, dim=1)
    return _BIN_CACHE[key]


def _quantiles(flat: torch.Tensor, qs: Sequence[float]) -> torch.Tensor:
    srt = flat.sort(dim=1)[0]
    idx = torch.tensor([int(q * (flat.shape[1] - 1)) for q in qs], device=flat.device)
    return srt[:, idx]


@torch.no_grad()
def summary_features(x: torch.Tensor) -> torch.Tensor:
    """Per-image summary statistics, shape ``(B, N_FEATURES)``.

    Contents: pixel-value quantiles, gradient-magnitude quantiles along both gate
    axes, a radial power spectrum, the histogram of row-to-row shift lags (sensitive
    to charge jumps) and statistics of the row/column means (drifts).
    """
    x = x[:, 0].float()
    b, h, w = x.shape
    f_val = _quantiles(x.flatten(1), (0.05, 0.25, 0.5, 0.75, 0.95))

    gx = (x[:, :, 1:] - x[:, :, :-1]).abs().flatten(1)
    gy = (x[:, 1:, :] - x[:, :-1, :]).abs().flatten(1)
    f_grad = torch.cat([_quantiles(gx, (0.5, 0.9, 0.99)), _quantiles(gy, (0.5, 0.9, 0.99))], dim=1)

    spec = torch.fft.fftshift(torch.fft.fft2(x - x.mean((1, 2), keepdim=True)).abs() ** 2, dim=(1, 2))
    f_spec = torch.log(spec.flatten(1) @ _radial_bins(h, w, x.device) + 1e-6)

    rows = x - x.mean(2, keepdim=True)
    corr = torch.stack(
        [(rows[:, :-1] * torch.roll(rows[:, 1:], l, dims=2)).mean(2) for l in (-2, -1, 0, 1, 2)], dim=2
    )  # (B, h-1, 5)
    lag = torch.nn.functional.one_hot(corr.argmax(2), 5).float().mean(1)

    rm, cm = x.mean(2), x.mean(1)
    f_drift = torch.stack([(rm[:, 1:] - rm[:, :-1]).abs().mean(1), (cm[:, 1:] - cm[:, :-1]).abs().mean(1)], dim=1)
    return torch.cat([f_val, f_grad, f_spec, lag, f_drift], dim=1)


def feature_distance(sim: torch.Tensor, tgt_mean: torch.Tensor, tgt_std: torch.Tensor) -> float:
    """Standardised distance between simulated feature statistics and target statistics."""
    f = summary_features(sim)
    scale = torch.maximum(tgt_std, 0.05 * tgt_mean.abs() + 1e-3)
    d_mean = ((f.mean(0) - tgt_mean) / scale) ** 2
    d_std = ((f.std(0) - tgt_std) / scale) ** 2
    return float((d_mean.sum() + 0.5 * d_std.sum()).item())


class RandomFeatureFD:
    """Frechet distance between image sets in a fixed random-CNN feature space (scaled by 1000)."""

    def __init__(self, device: str, seed: int = 123):
        g = torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
        ).to(device).eval()
        self.device = device
        del g

    @torch.no_grad()
    def _stats(self, x: torch.Tensor):
        feats = []
        for i in range(0, len(x), 128):
            z = self.net(x[i : i + 128].to(self.device))
            feats.append(torch.cat([z.mean((2, 3)), z.std((2, 3))], dim=1))
        f = torch.cat(feats).double().cpu().numpy()
        return f.mean(0), np.cov(f, rowvar=False) + 1e-6 * np.eye(f.shape[1])

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> float:
        m1, c1 = self._stats(a)
        m2, c2 = self._stats(b)
        covmean = linalg.sqrtm(c1 @ c2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return 1e3 * float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * covmean))
