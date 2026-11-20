"""Symmetric InfoNCE with configurable negative weighting."""

import numpy as np
import torch
import torch.nn.functional as F


def minmax_params(points):
    """Min-max normalisation of (m_Phi, Lambda) over the available grid."""
    p = np.asarray(points, dtype=np.float64)
    lo = p.min(0)
    hi = p.max(0)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    return lo, rng


def standardise_params(points, ref_points=None):
    """Standardisation statistics of (m_Phi, Lambda) taken from ``ref_points``."""
    ref = np.asarray(ref_points if ref_points is not None else points, dtype=np.float64)
    mu = ref.mean(0)
    sd = ref.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def param_matrix(points, mu, sd):
    p = np.asarray(points, dtype=np.float64)
    return (p - mu) / sd


def close_mask(points, dmass=0.0, dlam=1.0):
    """Boolean matrix: True where two points are too close to be used as negatives.

    Two points are 'close' when their mediator masses agree to within ``dmass`` GeV and
    their confinement scales differ by at most ``dlam``.
    """
    p = np.asarray(points, dtype=np.float64)
    dm = np.abs(p[:, None, 0] - p[None, :, 0])
    dl = np.abs(p[:, None, 1] - p[None, :, 1])
    m = (dm <= dmass) & (dl <= dlam)
    np.fill_diagonal(m, False)
    return m


def distance_weights(points, mu, sd, d0=1.0):
    """Weight of a negative as a function of its parameter distance from the anchor.

    ``w(d) = min(1, (d/d0)^2)``: nearby parameter points are down-weighted (they are
    too similar to be forced apart), distant ones keep the full negative strength.
    """
    z = param_matrix(points, mu, sd)
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    W = np.minimum(1.0, (d / d0) ** 2)
    np.fill_diagonal(W, 1.0)
    return W


def build_negative_weights(strategy, points, mu, sd, dmass=0.0, dlam=1.0, d0=1.0, dev="cpu"):
    """Return (W, allowed).

    ``W[p, q]``  weight of point ``q`` as a negative of point ``p`` (diagonal is 1 and
                 carries the positive pair, which is never down-weighted).
    ``allowed[p, q]``  True if ``q`` may be used as a negative of ``p``.
    """
    n = len(points)
    if strategy == "standard":
        W = np.ones((n, n), dtype=np.float64)
    elif strategy == "exclude_nearby":
        W = np.ones((n, n), dtype=np.float64)
        W[close_mask(points, dmass, dlam)] = 0.0
    elif strategy == "distance_weighted":
        W = distance_weights(points, mu, sd, d0)
    else:
        raise ValueError(f"unknown negative strategy {strategy}")
    np.fill_diagonal(W, 1.0)
    Wt = torch.tensor(W, dtype=torch.float32, device=dev)
    return Wt, (Wt > 0)


def event_prototype_loss(ge_a, ge_b, n_points, k, tau=0.1):
    """Per-event contrastive loss against per-point prototypes.

    ``ge_a``, ``ge_b`` are the unit per-event projections of the two views,
    shape (n_points * k, d), laid out point-major.  Each point's prototype is
    the mean of its own events in the other view; every event is classified
    against those prototypes.  No labels enter beyond the point identity.
    """
    d = ge_a.shape[-1]
    pa = F.normalize(ge_a.view(n_points, k, d).mean(1), dim=-1)
    pb = F.normalize(ge_b.view(n_points, k, d).mean(1), dim=-1)
    labels = torch.arange(n_points, device=ge_a.device).repeat_interleave(k)
    logits_a = (ge_a @ pb.T) / tau
    logits_b = (ge_b @ pa.T) / tau
    loss = 0.5 * (F.cross_entropy(logits_a, labels) + F.cross_entropy(logits_b, labels))
    with torch.no_grad():
        acc = 0.5 * (float((logits_a.argmax(1) == labels).float().mean())
                     + float((logits_b.argmax(1) == labels).float().mean()))
    return loss, acc


def supcon_loss(emb, labels, tau=0.1):
    """Supervised contrastive loss: same-label events are positives."""
    z = F.normalize(emb, dim=-1)
    sim = (z @ z.T) / tau
    labels = labels.reshape(-1, 1)
    eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    pos = (labels == labels.T) & ~eye
    pos_logits = sim.masked_fill(~pos, float("-inf"))
    all_logits = sim.masked_fill(eye, float("-inf"))
    loss = (torch.logsumexp(all_logits, dim=1) - torch.logsumexp(pos_logits, dim=1)).mean()
    with torch.no_grad():
        pred = all_logits.argmax(1)
        acc = float((pred == labels.reshape(-1)).float().mean().item())
    return loss, acc


def mass_prototype_loss(h, q_proto, mass_id, tau=0.1):
    """Contrast each embedding against frozen, mass-averaged LLM prototypes."""
    logits = (h @ q_proto.T) / tau
    loss = F.cross_entropy(logits, mass_id)
    with torch.no_grad():
        acc = float((logits.argmax(1) == mass_id).float().mean().item())
    return loss, acc


def info_nce_two_view(z_a, z_b, tau, point_ids, neg_w=None, neg_allowed=None):
    """Symmetric two-view InfoNCE.

    z_a, z_b    (N, d) L2-normalised embeddings of the two views of every point
    point_ids   (N,)   index into the negative-weight table
    neg_w       (P, P) optional negative weights
    neg_allowed (P, P) optional boolean table of admissible negatives
    """
    N = z_a.shape[0]
    logits = (z_a @ z_b.T) / tau
    rows = torch.arange(N, device=z_a.device)
    eye = torch.eye(N, dtype=torch.bool, device=z_a.device)

    if neg_allowed is None:
        keep = torch.ones_like(eye)
    else:
        keep = eye | neg_allowed[point_ids[:, None], point_ids[None, :]]
    if neg_w is not None:
        w = neg_w[point_ids[:, None], point_ids[None, :]]
        logits = logits + torch.log(w.clamp_min(1e-30))
    logits = logits.masked_fill(~keep, -float("inf"))

    log_denom = torch.logsumexp(logits, dim=1)
    loss = (-(logits[rows, rows] - log_denom)).mean()

    with torch.no_grad():
        pred = logits.argmax(1)
        acc = float((pred == point_ids).float().mean().item())
        nneg = float((keep & ~eye).sum(1).float().mean().item())
    return loss, {"acc": acc, "n_neg": nneg}
