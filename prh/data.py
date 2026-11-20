"""Event store for the PRH study.

Two caches are supported:

``fisher_geometry/cache/prh_cache.pt``       truncated to 128 particles, as the
                                              production cache (all masks identically
                                              true, so the event multiplicity is lost).
``fisher_geometry/cache/prh_cache_full.pt``   the same 128 highest-pT particles plus
                                              eight per-event soft-spectrum summaries
                                              computed over the full reconstructed event.

The second one is used for the primary results: the production truncation removes the
event multiplicity and the whole soft part of the spectrum, which handicaps the detector
modality relative to a theory encoder that receives the exact parameters.
"""

import os

import numpy as np
import torch

from .splits import ALL_POINTS, GROUPS

DEFAULT_CACHE = "fisher_geometry/cache.pt"
DEFAULT_OUT = "fisher_geometry/cache/prh_cache.pt"
FULL_OUT = "fisher_geometry/cache/prh_cache_full.pt"


def _point_index(points):
    return {tuple(int(v) for v in p): i for i, p in enumerate(points)}


def build_cache(src=DEFAULT_CACHE, dst=DEFAULT_OUT, verbose=True):
    """Extract and standardise the events needed by the PRH study (truncated cache)."""
    if os.path.exists(dst):
        if verbose:
            print(f"[prh] cache already present: {dst}")
        return dst
    raw = torch.load(src, map_location="cpu", weights_only=False)
    points = [tuple(int(v) for v in p) for p in raw["points"]]
    pid_all = raw["pid"].long()
    split_all = raw["split"]
    n_feat = int(raw["x"].shape[-1])

    keep_points = set(ALL_POINTS)
    train_pts = set(GROUPS["train"])

    is_train0 = torch.zeros(len(pid_all), dtype=torch.bool)
    for p in sorted(train_pts):
        i = _point_index(points)[p]
        is_train0 |= (pid_all == i) & (split_all == 0)
    is_eval2 = torch.zeros(len(pid_all), dtype=torch.bool)
    for p in sorted(keep_points):
        i = _point_index(points)[p]
        is_eval2 |= (pid_all == i) & (split_all == 2)

    sel = is_train0 | is_eval2
    idx = sel.nonzero().squeeze(1)
    if verbose:
        print(f"[prh] selected {len(idx)} of {len(pid_all)} events")

    x = raw["x"][idx]
    mask = raw["mask"][idx]
    pid = pid_all[idx]
    split = split_all[idx]

    tr = torch.zeros(n_feat, dtype=torch.float64)
    tr2 = torch.zeros(n_feat, dtype=torch.float64)
    cnt = 0.0
    tri = is_train0[idx].nonzero().squeeze(1)
    for s in range(0, len(tri), 20000):
        j = tri[s:s + 20000]
        xb = x[j].double()
        mb = mask[j].unsqueeze(-1)
        tr += (xb * mb).sum((0, 1))
        tr2 += (xb * xb * mb).sum((0, 1))
        cnt += float(mb.sum())
    mean = (tr / cnt).float()
    std = (tr2 / cnt - (tr / cnt) ** 2).clamp_min(1e-12).sqrt().float()

    for s in range(0, len(x), 50000):
        j = slice(s, s + 50000)
        xb = (x[j].float() - mean) / std
        x[j] = (xb * mask[j].unsqueeze(-1)).half()

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(
        dict(x=x, mask=mask, pid=pid.to(torch.int16), split=split.to(torch.uint8),
             points=points, mean=mean, std=std, feat_names=list(raw["feat_names"])),
        dst + ".tmp",
    )
    os.replace(dst + ".tmp", dst)
    if verbose:
        print(f"[prh] wrote {dst}")
    return dst


class PrhData:
    """Event store for the PRH study.  Everything lives on the compute device."""

    def __init__(self, cache_path, dev, verbose=True):
        d = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.dev = dev
        self.points = [tuple(int(v) for v in p) for p in d["points"]]
        self.n_feat = int(d["x"].shape[-1])
        self.feat_names = list(d.get("feat_names", []))
        self.summ_names = list(d.get("summ_names", []))
        self.n_summ = len(self.summ_names)
        self.mean = d["mean"]
        self.std = d["std"]
        self.x = d["x"].to(dev)
        self.mask = d["mask"].to(dev)
        self.pid = d["pid"].to(dev).long()
        self.split = d["split"].to(dev)
        self.summ = d["summ"].to(dev) if self.n_summ else None
        self._pools = {}
        if verbose:
            print(f"[prh] loaded {self.x.shape[0]} events, n_feat={self.n_feat}, "
                  f"n_summ={self.n_summ}, {len(self.points)} points")

    def pool(self, point, split):
        """Event indices belonging to ``point`` in ``split``."""
        key = (tuple(point), int(split))
        if key not in self._pools:
            m = (self.pid == self._point_id(point)) & (self.split == int(split))
            self._pools[key] = m.nonzero().squeeze(1)
        return self._pools[key]

    def _point_id(self, point):
        return self.points.index(tuple(int(v) for v in point))

    def get(self, ev):
        """Return (x, mask) for the given event indices, on device."""
        return self.x[ev].float(), self.mask[ev]

    def get_summ(self, ev):
        return self.summ[ev] if self.summ is not None else None


class CursorSampler:
    """Samples disjoint event blocks per parameter point.

    For every point a random permutation of its event pool is kept together with a
    cursor.  Each call returns the next ``block`` events of each point.  Within a call
    the events of a point are disjoint whenever the pool is large enough, which is what
    the detector-side positive pairs require.
    """

    def __init__(self, data: PrhData, points, split, generator=None):
        self.data = data
        self.points = list(points)
        self.split = split
        self.gen = generator
        self.perm = []
        self.cursor = []
        for p in self.points:
            pool = data.pool(p, split)
            if len(pool) == 0:
                raise RuntimeError(f"empty pool for point {p} split {split}")
            self.perm.append(pool[torch.randperm(len(pool), generator=generator)])
            self.cursor.append(0)

    def block(self, n_per_point, per_point_ids=None):
        """Return an int64 tensor of shape (n_points, n_per_point) of event indices."""
        ids = range(len(self.points)) if per_point_ids is None else per_point_ids
        out = []
        for i in ids:
            pool = self.perm[i]
            n = len(pool)
            c = self.cursor[i]
            if c + n_per_point > n:
                pool = pool[torch.randperm(n, generator=self.gen)]
                self.perm[i] = pool
                c = 0
            out.append(pool[c:c + n_per_point])
            self.cursor[i] = c + n_per_point
        return torch.stack(out)


def sample_disjoint(data: PrhData, point, n_total, split, generator=None):
    """Sample ``n_total`` distinct events of ``point`` (without replacement if possible)."""
    pool = data.pool(point, split)
    n = len(pool)
    if n >= n_total:
        return pool[torch.randperm(n, generator=generator)[:n_total]]
    reps = (n_total + n - 1) // n
    return pool.repeat(reps)[:n_total]
