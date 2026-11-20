import glob
import hashlib
import json
import math
import os
import re
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

HELD_INTERP = [(300, 2), (300, 3), (400, 2), (400, 3)]
HELD_EXTRAP = [(550, 1), (550, 2), (550, 3), (550, 4)]
HELD = set(HELD_INTERP) | set(HELD_EXTRAP)

DIR_RE = re.compile(r"SUEPlike_HV_mPhi(\d+)_mX(\d+)_Lam(\d+)$")


def theta_of(points):
    p = np.asarray(points, dtype=np.float32)
    return np.stack([(p[:, 0] - 340.0) / 130.0, (np.log(p[:, 1]) - 0.8) / 0.5], 1).astype(np.float32)


def theta_to_raw(tn):
    tn = np.asarray(tn, dtype=np.float64)
    return np.stack([tn[..., 0] * 130.0 + 340.0, np.exp(tn[..., 1] * 0.5 + 0.8)], -1)


def dist_init():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dev = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=3))
    return rank, world, dev


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def autocast(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def _is_pt(n):
    return re.search(r"(^|[^a-z])pt([^a-z]|$)", n.lower()) is not None


def _is_phi(n):
    return re.search(r"(^|[^a-z])phi([^a-z]|$)", n.lower()) is not None


def _is_energy(n):
    return re.search(r"mass|energy|(^|[^a-z])e([^a-z]|$)", n.lower()) is not None


def _list_columns(tbl):
    import pyarrow as pa

    out = {}
    for name in tbl.column_names:
        col = tbl[name]
        t = col.type
        if not (pa.types.is_list(t) or pa.types.is_large_list(t)):
            continue
        arr = col.combine_chunks()
        off = arr.offsets.to_numpy().astype(np.int64)
        vt = t.value_type
        if pa.types.is_struct(vt):
            for i in range(vt.num_fields):
                f = vt.field(i)
                if pa.types.is_floating(f.type) or pa.types.is_integer(f.type):
                    v = arr.values.field(i).to_numpy(zero_copy_only=False).astype(np.float32)
                    out[f"{name}.{f.name}"] = (off, v)
        elif pa.types.is_floating(vt) or pa.types.is_integer(vt):
            v = arr.values.to_numpy(zero_copy_only=False).astype(np.float32)
            out[name] = (off, v)
    return out


def _choose_columns(cols, override, drop):
    if override:
        names = [c.strip() for c in override.split(",")]
        missing = [n for n in names if n not in cols]
        if missing:
            raise SystemExit(f"--cols not found: {missing}. available: {sorted(cols)}")
        return names
    groups = {}
    for n, (off, _) in cols.items():
        key = hashlib.md5(np.diff(off).tobytes()).hexdigest()
        groups.setdefault(key, []).append(n)
    best = max(groups.values(), key=lambda g: (any(_is_pt(n) for n in g), len(g)))
    if drop:
        best = [n for n in best if not any(d in n for d in drop.split(","))]
    return best


def _featurize(name, a):
    if _is_pt(name):
        return [("logpt", np.log(np.maximum(a, 1e-3)))]
    if _is_phi(name):
        return [("sinphi", np.sin(a)), ("cosphi", np.cos(a))]
    if _is_energy(name):
        return [(f"log1p_{name}", np.log1p(np.maximum(a, 0.0)))]
    return [(name, a)]


def _process_file(path, names, P):
    import pyarrow.parquet as pq

    tbl = pq.read_table(path)
    cols = _list_columns(tbl)
    if any(n not in cols for n in names):
        raise RuntimeError(f"{path}: missing columns")
    off0 = cols[names[0]][0]
    lens = np.diff(off0)
    for n in names[1:]:
        if not np.array_equal(np.diff(cols[n][0]), lens):
            raise RuntimeError(f"{path}: list lengths of {n} differ from {names[0]}")
    n_ev = len(lens)
    L = max(int(lens.max()), 1)
    idx = np.arange(L)[None, :]
    valid = idx < lens[:, None]
    dense = {}
    for n in names:
        off, v = cols[n]
        g = np.minimum(off[:-1, None] + idx, max(len(v) - 1, 0))
        d = np.nan_to_num(v[g] if len(v) else np.zeros((n_ev, L), np.float32))
        dense[n] = np.where(valid, d, 0.0).astype(np.float32)
    pt_name = next((n for n in names if _is_pt(n)), None)
    if pt_name is not None:
        key = np.where(valid, dense[pt_name], -np.inf)
        order = np.argsort(-key, axis=1, kind="stable")
        Lm = min(L, P)
        order = order[:, :Lm]
    else:
        Lm = min(L, P)
        order = np.tile(np.arange(Lm)[None, :], (n_ev, 1))
    vsel = np.take_along_axis(valid, order, 1)
    feats, fnames = [], []
    for n in names:
        a = np.take_along_axis(dense[n], order, 1)
        for fn, fa in _featurize(n, a):
            feats.append(np.where(vsel, fa, 0.0).astype(np.float32))
            fnames.append(fn)
    x = np.stack(feats, -1)
    out = np.zeros((n_ev, P, x.shape[-1]), np.float16)
    m = np.zeros((n_ev, P), bool)
    out[:, :Lm] = x
    m[:, :Lm] = vsel
    return out, m, fnames


def build_cache(root, path, P, cols_override=None, drop=None, seed=0, max_files=None):
    import pyarrow.parquet as pq

    dirs = []
    for d in sorted(glob.glob(os.path.join(root, "SUEPlike_HV_mPhi*_mX*_Lam*"))):
        mt = DIR_RE.search(os.path.basename(d))
        if mt and os.path.isdir(d):
            dirs.append((int(mt.group(1)), int(mt.group(3)), d))
    dirs.sort()
    if not dirs:
        raise SystemExit(f"no SUEPlike_HV_mPhi*_mX*_Lam* directories under {root}")
    first = sorted(glob.glob(os.path.join(dirs[0][2], "*.parquet")))[0]
    print("schema of", first, flush=True)
    print(pq.read_schema(first), flush=True)
    names = _choose_columns(_list_columns(pq.read_table(first)), cols_override, drop)
    print("particle columns used:", names, flush=True)
    xs, ms, pids, splits, points, fnames = [], [], [], [], [], None
    for pi, (mphi, lam, d) in enumerate(dirs):
        files = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if max_files:
            files = files[:max_files]
        px, pm = [], []
        for f in files:
            x, m, fnames = _process_file(f, names, P)
            px.append(x)
            pm.append(m)
        px, pm = np.concatenate(px), np.concatenate(pm)
        n = len(px)
        perm = np.random.RandomState(seed + pi).permutation(n)
        sp = np.zeros(n, np.uint8)
        sp[perm[int(0.8 * n):int(0.9 * n)]] = 1
        sp[perm[int(0.9 * n):]] = 2
        xs.append(px)
        ms.append(pm)
        pids.append(np.full(n, pi, np.int16))
        splits.append(sp)
        points.append((mphi, lam))
        print(f"  mPhi={mphi} Lam={lam}: {n} events, mean multiplicity {pm.sum(1).mean():.1f}", flush=True)
    torch.save(
        dict(
            x=torch.from_numpy(np.concatenate(xs)),
            mask=torch.from_numpy(np.concatenate(ms)),
            pid=torch.from_numpy(np.concatenate(pids)),
            split=torch.from_numpy(np.concatenate(splits)),
            points=points,
            feat_names=fnames,
            cols=names,
        ),
        path + ".tmp",
    )
    os.replace(path + ".tmp", path)


def ensure_cache(args, rank):
    path = args.cache
    if not os.path.exists(path) and rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        build_cache(args.data_root, path, args.max_particles, args.cols, args.drop_cols)
    barrier()
    return path


def add_data_args(ap):
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--cache", default="fisher_geometry/cache.pt")
    ap.add_argument("--max-particles", type=int, default=128)
    ap.add_argument("--cols", default=None)
    ap.add_argument("--drop-cols", default=None)


class Data:
    def __init__(self, path, dev):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.dev = dev
        self.points = [tuple(p) for p in d["points"]]
        self.feat_names = d["feat_names"]
        self.n_feat = d["x"].shape[-1]
        self.x = d["x"].to(dev)
        self.mask = d["mask"].to(dev)
        self.pid = d["pid"].to(dev).long()
        self.split = d["split"].to(dev)
        held_ids = torch.tensor([i for i, p in enumerate(self.points) if p in HELD], device=dev)
        use = (self.split == 0) & ~torch.isin(self.pid, held_ids)
        s1 = torch.zeros(self.n_feat, dtype=torch.float64, device=dev)
        s2 = torch.zeros_like(s1)
        cnt = 0.0
        idx = use.nonzero().squeeze(1)
        for i in range(0, len(idx), 20000):
            j = idx[i:i + 20000]
            xb = self.x[j].double()
            mb = self.mask[j].unsqueeze(-1)
            s1 += (xb * mb).sum((0, 1))
            s2 += (xb * xb * mb).sum((0, 1))
            cnt += float(mb.sum())
        self.mean = (s1 / cnt).float()
        self.std = (s2 / cnt - (s1 / cnt) ** 2).clamp_min(1e-12).sqrt().float()
        for i in range(0, len(self.x), 50000):
            xb = (self.x[i:i + 50000].float() - self.mean) / self.std
            self.x[i:i + 50000] = (xb * self.mask[i:i + 50000].unsqueeze(-1)).half()
        self.theta = torch.tensor(theta_of(self.points), device=dev)
        self._ev = {}
        names = [n.lower() for n in self.feat_names]
        self.sin_i = [i for i, n in enumerate(names) if n == "sinphi"]
        self.cos_i = [i for i, n in enumerate(names) if n == "cosphi"]
        self.eta_i = [i for i, n in enumerate(names) if re.search(r"(^|[^a-z])eta([^a-z]|$)", n)]
        if len(self.sin_i) != len(self.cos_i):
            self.sin_i, self.cos_i = [], []

    def events(self, p, split):
        key = (p, split)
        if key not in self._ev:
            self._ev[key] = ((self.pid == p) & (self.split == split)).nonzero().squeeze(1)
        return self._ev[key]

    def augment(self, x, m):
        B = len(x)
        xr = x * self.std + self.mean
        if self.sin_i:
            a = torch.rand(B, 1, device=x.device) * 2 * math.pi
            ca, sa = a.cos(), a.sin()
            for si, ci in zip(self.sin_i, self.cos_i):
                s_, c_ = xr[..., si], xr[..., ci]
                ns, nc = s_ * ca + c_ * sa, c_ * ca - s_ * sa
                xr[..., si], xr[..., ci] = ns, nc
        if self.eta_i:
            flip = (torch.rand(B, 1, device=x.device) < 0.5).float() * 2 - 1
            for ei in self.eta_i:
                xr[..., ei] = xr[..., ei] * flip
        return ((xr - self.mean) / self.std) * m.unsqueeze(-1)

    def get(self, ev, aug=False):
        m = self.mask[ev]
        n = max(int(m.sum(1).max()), 1)
        n = min(((n + 15) // 16) * 16, m.shape[1])
        x, m = self.x[ev][:, :n].float(), m[:, :n]
        if aug:
            x = self.augment(x, m)
        return x, m


class Sampler:
    def __init__(self, data, pts, split):
        self.dev = data.dev
        lists = [data.events(p, split) for p in pts]
        self.cnt = torch.tensor([len(l) for l in lists], device=self.dev)
        self.tab = torch.zeros(len(pts), int(self.cnt.max()), dtype=torch.long, device=self.dev)
        for i, l in enumerate(lists):
            self.tab[i, : len(l)] = l

    def events(self, k_idx):
        r = (torch.rand(len(k_idx), device=self.dev) * self.cnt[k_idx]).long()
        return self.tab[k_idx, r]


class Block(nn.Module):
    def __init__(self, d, h, r=4, p=0.0):
        super().__init__()
        self.h = h
        self.p = p
        self.dr = nn.Dropout(p)
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, r * d), nn.GELU(), nn.Linear(r * d, d))

    def forward(self, x, am):
        B, S, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=self.p if self.training else 0.0)
        x = x + self.dr(self.o(a.transpose(1, 2).reshape(B, S, D)))
        return x + self.dr(self.ff(self.n2(x)))


class PointTransformer(nn.Module):
    def __init__(self, n_feat, d=256, L=8, h=8, p=0.0):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, h, 4, p) for _ in range(L)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask, cond=None):
        B = len(x)
        toks = [self.cls.expand(B, -1, -1)]
        ok = [torch.ones(B, 1, dtype=torch.bool, device=x.device)]
        if cond is not None:
            toks.append(cond.unsqueeze(1))
            ok.append(torch.ones(B, 1, dtype=torch.bool, device=x.device))
        toks.append(self.emb(x))
        ok.append(mask)
        h = torch.cat(toks, 1)
        am = torch.cat(ok, 1)[:, None, None, :]
        for b in self.blocks:
            h = b(h, am)
        return self.norm(h[:, 0])


class RatioNet(nn.Module):
    def __init__(self, n_feat, d=256, L=8, h=8, p=0.0):
        super().__init__()
        self.backbone = PointTransformer(n_feat, d, L, h, p)
        self.theta = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Linear(d, 1)

    def forward(self, x, mask, theta):
        return self.head(self.backbone(x, mask, self.theta(theta))).squeeze(-1)


class Joint(nn.Module):
    def __init__(self, n_feat, d=256, L=8, h=8, dlat=16, p=0.0):
        super().__init__()
        self.psi = PointTransformer(n_feat, d, L, h, p)
        self.head = nn.Linear(d, dlat)
        self.phi = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, dlat)
        )

    def event_embed(self, x, mask):
        return self.head(self.psi(x, mask))

    def forward(self, x, mask, B, k, theta_all):
        z = self.event_embed(x, mask).view(B, k, -1)
        return z.mean(1), self.phi(theta_all)


def knn_geodesic(D, theta, ids, k=6):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    n = len(ids)
    t = theta[ids]
    dd = np.linalg.norm(t[:, None] - t[None], axis=-1)
    W = np.zeros((n, n))
    for i in range(n):
        nb = np.argsort(dd[i])[1:k + 1]
        for j in nb:
            w = D[ids[i], ids[j]]
            W[i, j] = W[j, i] = w
    sp = shortest_path(csr_matrix(W), directed=False)
    out = np.full(D.shape, np.nan)
    out[np.ix_(ids, ids)] = sp
    return out


def cosine_lr(step, total, warm, base):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))