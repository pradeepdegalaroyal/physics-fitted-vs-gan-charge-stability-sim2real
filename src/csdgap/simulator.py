"""Constant-interaction double-dot simulator with a parametric measurement layer.

The simulator produces charge stability diagrams (CSDs) of a double quantum dot
as seen through a capacitively coupled charge sensor, together with an exact
mask of the charge-transition lines.  A flat parameter vector ``theta`` controls
both the physics (inter-dot coupling, gate cross-talk) and the measurement chain
(thermal broadening, sensor non-linearity, drifts, charge jumps).  The *source*
domain is a clean textbook simulation; the *target* domain is a harsher proxy of
a laboratory measurement.  No experimental data is used anywhere in this package.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

NMAX = 6  # occupation numbers 0..NMAX per dot
NS = NMAX + 1

PARAM_NAMES: Tuple[str, ...] = (
    "temperature",  # thermal broadening of transitions (energy units of E_C1)
    "sigma_white",  # white sensor noise (fraction of signal range)
    "sigma_pink",  # 1/f raster-time drift amplitude
    "peak_mix",  # weight of Coulomb-peak (non-linear) sensor response
    "peak_center",  # position of the sensor peak on the normalised signal
    "peak_width",  # half width of the sensor peak
    "jump_prob",  # probability per scan line that a charge trap toggles
    "jump_amp",  # voltage-offset size of a charge jump (transition units)
    "coupling_mean",  # mean inter-dot mutual charging energy / E_C1
    "crosstalk_mean",  # mean gate cross-talk (lever-arm skew)
    "bg_slope",  # background tilt of the sensor signal
)

# source: clean textbook simulation.  target: laboratory-like proxy.
THETA_SOURCE = np.array([0.02, 0.02, 0.0, 0.0, 0.5, 0.4, 0.0, 0.3, 0.20, 0.04, 0.0])
THETA_TARGET = np.array([0.09, 0.06, 0.05, 0.7, 0.45, 0.25, 0.04, 0.35, 0.35, 0.20, 0.25])
BOUNDS = np.array(
    [
        [0.01, 0.25],
        [0.005, 0.2],
        [0.0, 0.2],
        [0.0, 1.0],
        [0.1, 0.9],
        [0.1, 1.0],
        [0.0, 0.15],
        [0.05, 0.6],
        [0.05, 0.6],
        [0.0, 0.35],
        [-0.5, 0.5],
    ]
)

# components of the measurement layer that the ablation can switch off
COMPONENTS = {
    "thermal broadening": (0,),
    "sensor peak": (3, 4, 5),
    "1/f drift": (2,),
    "charge jumps": (6, 7),
    "physics shift": (8, 9),
    "background tilt": (10,),
}

TARGET_ASYMMETRY = 0.6  # lever-arm asymmetry present in the target but not in the model


def _uniform(gen: torch.Generator, lo: float, hi: float, shape: Tuple[int, ...], device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)


def _pink_noise(n: int, length: int, gen: torch.Generator, device) -> torch.Tensor:
    """Unit-variance 1/f noise in raster time, shape (n, length)."""
    white = torch.randn(n, length, generator=gen, device=device)
    spec = torch.fft.rfft(white, dim=1)
    freq = torch.arange(spec.shape[1], device=device, dtype=torch.float32).clamp(min=1.0)
    spec = spec / freq.sqrt()
    spec[:, 0] = 0
    out = torch.fft.irfft(spec, n=length, dim=1)
    return out / out.std(dim=1, keepdim=True).clamp(min=1e-8)


@torch.no_grad()
def _render_chunk(
    th: np.ndarray,
    n: int,
    size: int,
    gen: torch.Generator,
    device,
    artefact: float,
    asym: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t_, sw, sp, lam, s0, w, pj, amp, kap_m, cr_m, bg = [float(v) for v in th]
    ar = torch.linspace(0, 1, size, device=device)
    x2 = ar.view(1, size, 1)  # rows: slow gate
    x1 = ar.view(1, 1, size)  # columns: fast gate

    s = _uniform(gen, 3.0, 4.2, (n,), device)[:, None, None]
    q = _uniform(gen, 0.85, 1.15, (n,), device)[:, None, None]
    r = _uniform(gen, 0.85, 1.15, (n,), device)
    kappa = (kap_m + _uniform(gen, -0.1, 0.1, (n,), device)).clamp(0.02, 0.7)
    cross = (cr_m + _uniform(gen, -0.06, 0.06, (n,), device)).clamp(min=0.0)[:, None, None]
    o1 = _uniform(gen, 0.0, 1.0, (n,), device)[:, None, None]
    o2 = _uniform(gen, 0.0, 1.0, (n,), device)[:, None, None]

    # telegraph charge jumps: one toggle chance per scan line (slow axis)
    flips = (torch.rand(n, size, generator=gen, device=device) < pj).long()
    start = torch.randint(0, 2, (n, 1), generator=gen, device=device)
    delta = ((flips.cumsum(1) + start) % 2).float() * amp
    delta = delta * (pj > 0)

    e1 = s * (x1 + cross * x2) + o1 + delta[:, :, None]
    e2 = s * q * (x2 + asym * cross * x1) + o2 + 0.6 * delta[:, :, None]

    ni = torch.arange(NS, device=device, dtype=torch.float32)
    n1 = ni.view(1, NS, 1)
    n2 = ni.view(1, 1, NS)
    e_const = 0.5 * n1**2 + 0.5 * r.view(n, 1, 1) * n2**2 + kappa.view(n, 1, 1) * n1 * n2  # (n, NS, NS)
    u = (
        e_const[:, :, :, None, None]
        - ni.view(1, NS, 1, 1, 1) * e1[:, None, None]
        - ni.view(1, 1, NS, 1, 1) * e2[:, None, None]
    ).flatten(1, 2)  # (n, NS*NS, size, size)

    # exact ground-state labels and thermally broadened occupation
    idx = u.argmin(1)
    edge = torch.zeros(n, 1, size, size, device=device)
    edge[:, 0, :, :-1] = torch.maximum(edge[:, 0, :, :-1], (idx[:, :, 1:] != idx[:, :, :-1]).float())
    edge[:, 0, :-1, :] = torch.maximum(edge[:, 0, :-1, :], (idx[:, 1:, :] != idx[:, :-1, :]).float())
    mask = torch.nn.functional.max_pool2d(edge, 3, stride=1, padding=1)

    temp = t_ * _uniform(gen, 0.85, 1.15, (n, 1, 1, 1), device)
    p = torch.softmax(-u / temp, dim=1)
    k = torch.arange(NS * NS, device=device)
    occ1 = (p * (k // NS).float().view(1, -1, 1, 1)).sum(1)
    occ2 = (p * (k % NS).float().view(1, -1, 1, 1)).sum(1)

    g1 = _uniform(gen, 0.8, 1.2, (n,), device)[:, None, None]
    g2 = _uniform(gen, 0.5, 0.9, (n,), device)[:, None, None]
    sig = occ1 * g1 + occ2 * g2
    flat = sig.flatten(1)
    lo, hi = flat.min(1)[0].view(n, 1, 1), flat.max(1)[0].view(n, 1, 1)
    sig = (sig - lo) / (hi - lo).clamp(min=1e-6)
    sig = sig + bg * (0.6 * x1 + 0.4 * x2)
    peak = 1.0 / (1.0 + ((sig - s0) / w) ** 2)
    sig = (1 - lam) * sig + lam * peak

    if artefact > 0:  # effects the fitted model does not contain
        fx = _uniform(gen, 2.0, 6.0, (n, 1, 1), device)
        fy = _uniform(gen, -3.0, 3.0, (n, 1, 1), device)
        ph = _uniform(gen, 0.0, 6.2832, (n, 1, 1), device)
        sig = sig + 0.25 * artefact * torch.sin(6.2832 * (fx * x1 + fy * x2) + ph)
        row0 = torch.randint(size // 4, 3 * size // 4, (n, 1, 1), generator=gen, device=device)
        rows = torch.arange(size, device=device).view(1, size, 1)
        active = (rows >= row0).float() * (torch.rand(n, 1, 1, generator=gen, device=device) < artefact).float()
        sgn = torch.where(torch.rand(n, 1, 1, generator=gen, device=device) < 0.5, -1.0, 1.0)
        sig = sig * (1 + 0.4 * artefact * sgn * active) + 0.3 * artefact * sgn * active

    sig = sig + sw * torch.randn(n, size, size, generator=gen, device=device)
    if sp > 0:
        sig = sig + sp * _pink_noise(n, size * size, gen, device).view(n, size, size)

    flat = sig.flatten(1)
    sig = (sig - flat.mean(1).view(n, 1, 1)) / flat.std(1).view(n, 1, 1).clamp(min=1e-6)
    img = (sig.clamp(-4, 4) / 4).unsqueeze(1)
    return img, mask


def render(
    theta: Sequence[float],
    n: int,
    size: int = 64,
    seed: int = 0,
    device: Optional[str] = None,
    artefact: float = 0.0,
    asym: float = 1.0,
    chunk: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render ``n`` charge stability diagrams and exact transition masks.

    Args:
        theta: parameter vector ordered as ``PARAM_NAMES``.
        n: number of images.
        size: image side length in pixels.
        seed: seed of the random generator (common random numbers for equal seeds).
        device: torch device string; defaults to CUDA when available.
        artefact: strength of unmodelled artefacts (interference stripes, sensor re-tuning).
        asym: lever-arm asymmetry (1 = symmetric, as in the fitted model).
        chunk: images per GPU batch.

    Returns:
        ``(images, masks)`` of shape ``(n, 1, size, size)``; images lie in [-1, 1].
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    th = np.asarray(theta, dtype=np.float64)
    imgs, masks = [], []
    for i in range(0, n, chunk):
        a, b = _render_chunk(th, min(chunk, n - i), size, gen, dev, artefact, asym)
        imgs.append(a)
        masks.append(b)
    return torch.cat(imgs), torch.cat(masks)


def render_source(n: int, size: int = 64, seed: int = 0, device: Optional[str] = None):
    """Clean textbook simulation (the training domain)."""
    return render(THETA_SOURCE, n, size, seed, device)


def render_target(n: int, misspec: float, size: int = 64, seed: int = 0, device: Optional[str] = None):
    """Laboratory-like proxy domain; ``misspec`` scales effects outside the fitted model family."""
    return render(THETA_TARGET, n, size, seed, device, artefact=misspec, asym=TARGET_ASYMMETRY)
