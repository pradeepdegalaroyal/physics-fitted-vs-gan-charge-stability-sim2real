"""Physics-structured translator: fit the simulator's parameters to unlabeled target images."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import differential_evolution

from .features import feature_distance, summary_features
from .simulator import BOUNDS, COMPONENTS, THETA_SOURCE, render


def free_indices(frozen: Sequence[str] = ()) -> np.ndarray:
    """Indices of parameters that are fitted, after freezing the named components."""
    off = {i for name in frozen for i in COMPONENTS[name]}
    return np.array([i for i in range(len(THETA_SOURCE)) if i not in off])


def fit_simulator(
    target: torch.Tensor,
    seed: int = 0,
    frozen: Sequence[str] = (),
    n_sim: int = 32,
    maxiter: int = 15,
    popsize: int = 8,
    size: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """Fit simulator parameters by matching feature statistics of unlabeled target images.

    Frozen components keep their source-domain value (this is the ablation switch).
    Evaluations use common random numbers, so the objective is deterministic.

    Returns:
        ``(theta_fit, final_objective)``.
    """
    size = size or target.shape[-1]
    dev = str(target.device) if device is None else device
    f_t = summary_features(target)
    t_mean, t_std = f_t.mean(0), f_t.std(0) if len(f_t) > 1 else torch.zeros_like(f_t[0])
    idx = free_indices(frozen)
    base = THETA_SOURCE.copy()

    def objective(z: np.ndarray) -> float:
        th = base.copy()
        th[idx] = z
        sim, _ = render(th, n_sim, size, seed=12345, device=dev)
        return feature_distance(sim, t_mean, t_std)

    res = differential_evolution(
        objective,
        BOUNDS[idx],
        x0=base[idx],
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        tol=0.0,
        polish=False,
        init="sobol",
        updating="immediate",
    )
    theta = base.copy()
    theta[idx] = res.x
    return theta, float(res.fun)


def param_table(theta: np.ndarray) -> Dict[str, float]:
    from .simulator import PARAM_NAMES

    return {k: float(v) for k, v in zip(PARAM_NAMES, theta)}
