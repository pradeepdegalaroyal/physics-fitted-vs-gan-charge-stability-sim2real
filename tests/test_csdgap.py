import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csdgap import BOUNDS, COMPONENTS, THETA_SOURCE, THETA_TARGET, render, render_source, render_target  # noqa: E402
from csdgap.features import N_FEATURES, summary_features  # noqa: E402
from csdgap.fit import fit_simulator, free_indices  # noqa: E402
from csdgap.segmenter import SmallUNet, dice_score  # noqa: E402
from csdgap.translators import Generator, domain_randomise  # noqa: E402

torch.set_num_threads(1)


def test_render_shapes_range_and_determinism():
    x, m = render_source(4, 32, seed=3, device="cpu")
    assert x.shape == m.shape == (4, 1, 32, 32)
    assert x.min() >= -1 and x.max() <= 1
    assert set(m.unique().tolist()) <= {0.0, 1.0}
    assert 0.02 < m.mean() < 0.9  # transition lines exist but do not fill the image
    x2, _ = render_source(4, 32, seed=3, device="cpu")
    assert torch.equal(x, x2)


def test_target_is_harder_than_source():
    xs, _ = render_source(16, 32, seed=0, device="cpu")
    xt, _ = render_target(16, 0.5, 32, seed=0, device="cpu")
    fs, ft = summary_features(xs), summary_features(xt)
    assert fs.shape == (16, N_FEATURES) and torch.isfinite(ft).all()
    # target has stronger low-frequency drift and broader transitions than the clean source
    assert (fs.mean(0) - ft.mean(0)).abs().sum() > 1.0


def test_charge_jumps_move_the_labels():
    th = THETA_SOURCE.copy()
    th[6] = 0.5  # toggle a charge trap on about half of the scan lines
    _, m_jump = render(th, 4, 32, seed=1, device="cpu")
    _, m_flat = render(THETA_SOURCE, 4, 32, seed=1, device="cpu")
    assert not torch.equal(m_jump, m_flat)


def test_parameter_bounds_contain_both_domains():
    for th in (THETA_SOURCE, THETA_TARGET):
        assert ((th >= BOUNDS[:, 0] - 1e-9) & (th <= BOUNDS[:, 1] + 1e-9)).all()


def test_frozen_components_stay_at_source_value():
    target, _ = render_target(6, 0.0, 32, seed=2, device="cpu")
    theta, obj = fit_simulator(target, seed=0, frozen=("charge jumps",), n_sim=4, maxiter=1, popsize=3, device="cpu")
    for i in COMPONENTS["charge jumps"]:
        assert theta[i] == THETA_SOURCE[i]
    assert np.isfinite(obj) and len(free_indices(("charge jumps",))) == len(THETA_SOURCE) - 2


def test_unet_and_dice():
    net = SmallUNet().eval()
    x = torch.randn(2, 1, 32, 32)
    assert net(x).shape == (2, 1, 32, 32)

    class Perfect(torch.nn.Module):
        def forward(self, z):
            return torch.where(self.mask > 0, 10.0, -10.0)

    mask = (torch.rand(3, 1, 16, 16) > 0.7).float()
    p = Perfect()
    p.mask = mask
    assert dice_score(p, torch.zeros_like(mask), mask) > 0.99


def test_translator_modules_keep_shape_and_range():
    x, _ = render_source(3, 32, seed=0, device="cpu")
    assert Generator()(x).shape == x.shape
    a = domain_randomise(x, 0)
    assert a.shape == x.shape and a.abs().max() <= 1
