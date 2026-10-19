import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLUB(nn.Module):
    """CLUB upper-bound estimator of I(f_a; f_o).

    q_phi(f_a | f_o) is a diagonal Gaussian. logvar is tanh-clamped to keep
    training stable (the proposal explicitly calls this out).
    """

    def __init__(self, dim, hidden=None, logvar_scale=4.0):
        super().__init__()
        h = hidden if hidden is not None else dim
        self.mu_net = nn.Sequential(
            nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, dim)
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, dim), nn.Tanh()
        )
        self.logvar_scale = logvar_scale

    def _gauss_logp(self, mu, logvar, x):
        return -((mu - x) ** 2) / (2.0 * logvar.exp()) - 0.5 * logvar

    def _mu_logvar(self, f_o):
        mu = self.mu_net(f_o)
        logvar = self.logvar_net(f_o) * self.logvar_scale
        return mu, logvar

    def forward(self, f_a, f_o):
        """Differentiable MI upper bound — minimize this in the main loss."""
        mu, logvar = self._mu_logvar(f_o)
        pos = self._gauss_logp(mu, logvar, f_a).sum(dim=-1)
        idx = torch.randperm(f_a.size(0), device=f_a.device)
        neg = self._gauss_logp(mu, logvar, f_a[idx]).sum(dim=-1)
        return (pos - neg).mean()

    def learning_loss(self, f_a, f_o):
        """NLL of q_phi(f_a | f_o); minimize to fit the variational approx."""
        mu, logvar = self._mu_logvar(f_o)
        return -self._gauss_logp(mu, logvar, f_a).sum(dim=-1).mean()


class FeatureBank:
    """Per-class deque memory bank for FACA. Features kept on CPU."""

    def __init__(self, num_classes, max_per_class=128):
        self.num_classes = num_classes
        self.max_per_class = max_per_class
        self.bank = [deque(maxlen=max_per_class) for _ in range(num_classes)]

    def update(self, feats, labels):
        feats = feats.detach().to('cpu', non_blocking=True)
        labels = labels.detach().to('cpu').tolist()
        for f, l in zip(feats, labels):
            self.bank[int(l)].append(f)

    def has(self, cls_idx):
        return len(self.bank[int(cls_idx)]) > 0

    def sample(self, cls_idx):
        return random.choice(self.bank[int(cls_idx)])

    def coverage(self):
        return sum(1 for b in self.bank if len(b) > 0) / max(self.num_classes, 1)


class ComposeFuser(nn.Module):
    """Light MLP fusing (f_a, f_o) into a synthetic compositional rep."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f_a, f_o):
        residual = 0.5 * (f_a + f_o)
        return residual + self.net(torch.cat([f_a, f_o], dim=-1))
