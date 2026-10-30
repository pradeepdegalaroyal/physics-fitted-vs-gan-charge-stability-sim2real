"""Image-space baselines: hand-tuned augmentation and an unpaired CycleGAN-style translator."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def domain_randomise(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Generic hand-tuned augmentation of clean images (no target data is used).

    Random Gaussian noise, Gaussian blur, a linear brightness ramp and a contrast gain.
    The ranges are fixed a priori and were not tuned on the target.
    """
    dev = x.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    n = len(x)
    out = torch.empty_like(x)
    sigma_blur = torch.rand(n, generator=gen, device=dev) * 2.0
    ks = torch.arange(-4, 5, device=dev).float()
    for i in range(n):  # per-image blur width; n is a few thousand at most and tensors are tiny
        s = sigma_blur[i].clamp(min=0.05)
        k = torch.exp(-(ks**2) / (2 * s**2))
        k = (k / k.sum()).view(1, 1, 1, -1)
        out[i : i + 1] = F.conv2d(F.conv2d(x[i : i + 1], k, padding=(0, 4)), k.transpose(2, 3), padding=(4, 0))
    size = x.shape[-1]
    ramp = torch.linspace(-1, 1, size, device=dev)
    ang = torch.rand(n, 1, 1, 1, generator=gen, device=dev) * 6.2832
    grad = torch.cos(ang) * ramp.view(1, 1, 1, -1) + torch.sin(ang) * ramp.view(1, 1, -1, 1)
    out = out * (0.7 + 0.6 * torch.rand(n, 1, 1, 1, generator=gen, device=dev))
    out = out + 0.5 * torch.rand(n, 1, 1, 1, generator=gen, device=dev) * grad / 4
    out = out + 0.4 / 4 * torch.rand(n, 1, 1, 1, generator=gen, device=dev) * torch.randn(
        x.shape, generator=gen, device=dev
    )
    return out.clamp(-1, 1)


class Generator(nn.Module):
    """Residual encoder-decoder; the output is the input plus a learned correction."""

    def __init__(self, ch: int = 24):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(True))
        self.down = nn.Sequential(
            nn.Conv2d(ch, 2 * ch, 3, 2, 1), nn.InstanceNorm2d(2 * ch), nn.ReLU(True),
            nn.Conv2d(2 * ch, 4 * ch, 3, 2, 1), nn.InstanceNorm2d(4 * ch), nn.ReLU(True),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(4 * ch, 4 * ch, 3, padding=1), nn.InstanceNorm2d(4 * ch), nn.ReLU(True),
            nn.Conv2d(4 * ch, 4 * ch, 3, padding=1), nn.InstanceNorm2d(4 * ch), nn.ReLU(True),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(4 * ch, 2 * ch, 3, padding=1), nn.InstanceNorm2d(2 * ch), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(2 * ch, ch, 3, padding=1), nn.ReLU(True),
        )
        self.out = nn.Conv2d(2 * ch, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.inp(x)
        h = self.up(self.mid(self.down(a)))
        return (x + self.out(torch.cat([h, a], 1))).clamp(-1, 1)


class PatchDiscriminator(nn.Module):
    """Patch-level critic for least-squares GAN training."""

    def __init__(self, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, 2 * ch, 4, 2, 1), nn.InstanceNorm2d(2 * ch), nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * ch, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_cyclegan(
    source: torch.Tensor,
    target: torch.Tensor,
    seed: int = 0,
    steps: int = 600,
    batch: int = 16,
    lam_cycle: float = 10.0,
) -> Generator:
    """Train an unpaired source-to-target translator (CycleGAN with least-squares loss).

    Only unlabeled target images are used; source masks are never seen by the GAN.
    """
    torch.manual_seed(seed)
    dev = source.device
    g_st, g_ts = Generator().to(dev), Generator().to(dev)
    d_t, d_s = PatchDiscriminator().to(dev), PatchDiscriminator().to(dev)
    opt_g = torch.optim.Adam(list(g_st.parameters()) + list(g_ts.parameters()), 2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(list(d_t.parameters()) + list(d_s.parameters()), 2e-4, betas=(0.5, 0.999))
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    mse = nn.MSELoss()
    for _ in range(steps):
        xs = source[torch.randint(0, len(source), (batch,), generator=gen, device=dev)]
        xt = target[torch.randint(0, len(target), (batch,), generator=gen, device=dev)]
        fake_t, fake_s = g_st(xs), g_ts(xt)
        pt, ps = d_t(fake_t), d_s(fake_s)
        loss_g = (
            mse(pt, torch.ones_like(pt)) + mse(ps, torch.ones_like(ps))
            + lam_cycle * (F.l1_loss(g_ts(fake_t), xs) + F.l1_loss(g_st(fake_s), xt))
        )
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        rt, rs, ft, fs = d_t(xt), d_s(xs), d_t(fake_t.detach()), d_s(fake_s.detach())
        loss_d = 0.5 * (
            mse(rt, torch.ones_like(rt)) + mse(ft, torch.zeros_like(ft))
            + mse(rs, torch.ones_like(rs)) + mse(fs, torch.zeros_like(fs))
        )
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()
    return g_st.eval()


@torch.no_grad()
def translate(g: Generator, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([g(x[i : i + 256]) for i in range(0, len(x), 256)])
