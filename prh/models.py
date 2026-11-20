"""Encoders for the PRH detector/theory study."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, width, depth, out_dim, dropout=0.0, act=nn.SiLU):
    layers = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = width
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TheoryMLP(nn.Module):
    """f_T : R^2 -> S^{d-1}.  Input is the min-max normalised (m_Phi, Lambda) pair."""

    def __init__(self, d=128, width=128, depth=3, dropout=0.0):
        super().__init__()
        self.net = mlp(2, width, depth, d, dropout)
        self.d = d

    def forward(self, theta):
        return F.normalize(self.net(theta), dim=-1)


class DeepSetsDetector(nn.Module):
    """f_D : set of reconstructed events -> S^{d-1}.

    Parameters
    ----------
    n_feat     number of per-particle features
    n_summ     number of per-event soft-spectrum summaries (0 to disable)
    backbone  "deepsets" (per-particle MLP) or "transformer" (point transformer)
    """

    def __init__(self, n_feat, d=128, width=128, depth=2, pwidth=64, pool="meanstd",
                 dropout=0.0, n_summ=0, backbone="deepsets", tf_d=64, tf_layers=2,
                 tf_heads=4):
        super().__init__()
        self.pool_mode = pool
        self.backbone = backbone
        self.n_summ = n_summ
        if backbone == "deepsets":
            self.phi = mlp(n_feat, pwidth, 1, pwidth, dropout)
            n_summary = pwidth * (2 if pool == "meanstd" else 1) + 1 + n_summ
            self.rho = mlp(n_summary, width, depth, width, dropout)
        elif backbone == "transformer":
            from .common import PointTransformer

            self.psi = PointTransformer(n_feat, tf_d, tf_layers, tf_heads, dropout)
            n_summary = tf_d + n_summ
            self.rho = mlp(n_summary, width, depth, width, dropout)
        else:
            raise ValueError(f"unknown backbone {backbone}")
        self.g = mlp(width, width, 1, d, dropout)
        self.d = d

    def event_embeddings(self, x, mask, summ=None):
        """x: (B, S, F), mask: (B, S), summ: (B, n_summ) -> (B, width)."""
        w = mask.unsqueeze(-1).float()
        if self.backbone == "deepsets":
            h = self.phi(x)
            n = w.sum(1).clamp_min(1.0)
            mean = (h * w).sum(1) / n
            parts = [mean]
            if self.pool_mode == "meanstd":
                var = ((h * h) * w).sum(1) / n - mean ** 2
                parts.append(var.clamp_min(0.0).sqrt())
            parts.append(torch.log1p(mask.sum(1).float()).unsqueeze(-1))
            feat = torch.cat(parts, dim=-1)
        else:
            feat = self.psi(x, mask)
        if self.n_summ:
            feat = torch.cat([feat, summ], dim=-1)
        return self.rho(feat)

    def forward(self, x, mask, n_views, k, summ=None, return_events=False):
        """x, mask: (n_views * k, S, F) laid out view-major.  Returns (n_views, d)."""
        e = self.event_embeddings(x, mask, summ)
        pooled = e.view(n_views, k, -1).mean(1)
        z = F.normalize(self.g(pooled), dim=-1)
        if return_events:
            return z, e
        return z
