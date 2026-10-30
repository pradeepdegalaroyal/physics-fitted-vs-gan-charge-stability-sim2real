"""Downstream task: segment charge-transition lines with a small U-Net."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Three-level U-Net (about 0.12 M parameters) returning per-pixel logits."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.e1, self.e2, self.e3 = _block(1, base), _block(base, 2 * base), _block(2 * base, 4 * base)
        self.d2 = _block(4 * base + 2 * base, 2 * base)
        self.d1 = _block(2 * base + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        d = self.d2(torch.cat([F.interpolate(c, scale_factor=2), b], 1))
        d = self.d1(torch.cat([F.interpolate(d, scale_factor=2), a], 1))
        return self.out(d)


def train_segmenter(
    images: torch.Tensor,
    masks: torch.Tensor,
    seed: int = 0,
    steps: int = 400,
    batch: int = 32,
    lr: float = 3e-3,
) -> SmallUNet:
    """Train with binary cross-entropy plus soft Dice loss."""
    torch.manual_seed(seed)
    dev = images.device
    net = SmallUNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    pos_w = torch.tensor(3.0, device=dev)
    net.train()
    for _ in range(steps):
        i = torch.randint(0, len(images), (batch,), generator=gen, device=dev)
        logit = net(images[i])
        y = masks[i]
        p = torch.sigmoid(logit)
        dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w) + dice.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


@torch.no_grad()
def dice_score(net: SmallUNet, images: torch.Tensor, masks: torch.Tensor) -> float:
    """Mean per-image Dice coefficient at threshold 0.5."""
    scores = []
    for i in range(0, len(images), 256):
        pred = (torch.sigmoid(net(images[i : i + 256])) > 0.5).float()
        y = masks[i : i + 256]
        inter = (pred * y).sum((1, 2, 3))
        scores.append((2 * inter + 1e-3) / (pred.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1e-3))
    return float(torch.cat(scores).mean().item())
