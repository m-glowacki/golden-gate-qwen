"""Contrastive training for theory and detector encoders."""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .data import CursorSampler, PrhData, build_cache, sample_disjoint
from .infonce import (build_negative_weights, event_prototype_loss,
                          info_nce_two_view, mass_prototype_loss, minmax_params,
                          standardise_params, supcon_loss)
from .models import DeepSetsDetector, TheoryMLP
from .splits import ALL_POINTS, GROUPS


# ----------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------
def theory_cfg(**kw):
    cfg = dict(
        seed=0,
        steps=3000,
        lr=1e-3,
        tau=0.1,
        dlat=16,
        width=128,
        depth=3,
        dropout=0.2,
        neg="standard",         # standard | exclude_nearby | distance_weighted
        dmass=50.0,            # exclusion radius in GeV for exclude_nearby
        dlam=1.0,              # exclusion radius in Lambda units
        d0=1.0,                # weight scale for distance_weighted
        out="fisher_geometry/runs/prh_platonic",
    )
    cfg.update(kw)
    return cfg


def detector_cfg(**kw):
    cfg = dict(
        seed=0,
        steps=3000,
        lr=3e-4,
        tau=0.1,
        dlat=16,
        width=128,
        depth=2,
        pwidth=64,
        pool="meanstd",
        dropout=0.0,
        backbone="deepsets",   # deepsets | transformer
        use_summ=True,         # per-event soft-spectrum summaries
        tf_d=64,
        tf_layers=2,
        tf_heads=4,
        k=128,                 # events per view
        neg="standard",
        dmass=50.0,
        dlam=1.0,
        d0=1.0,
        phys_aug=False,        # azimuthal-rotation / eta-flip invariance
        evt_weight=0.0,        # weight of the per-event prototype loss
        evt_tau=0.1,
        sup_weight=0.0,        # supervised contrastive loss on per-event embeddings
        sup_tau=0.1,
        train_points=None,     # override GROUPS["train"] (mass holdout etc.)
        train_split=0,
        cm_weight=0.0,         # frozen-LLM cross-modal loss weight (set level)
        cm_evt_weight=0.0,     # frozen-LLM cross-modal loss weight (per event)
        cm_evt_align_weight=0.0,  # per-event cosine regression onto the raw card rep
        cm_mode="point",       # point | mass (contrast against mass prototypes)
        cm_tau=0.1,
        cm_dim=16,
        llm_npz="fisher_geometry/runs/prh_llm_qwen3_8b/llm_embeddings.npz",
        llm_variant="hv_only",
        llm_pooling="last",
        llm_layer=18,
        out="fisher_geometry/runs/prh_platonic",
    )
    cfg.update(kw)
    return cfg


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def dev_of():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda", 0)
    return torch.device("cpu")


def cosine_lr(step, total, warm, base):
    if step < warm:
        return base * (step + 1) / max(1, warm)
    p = (step - warm) / max(1, total - warm)
    return base * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))


def minmax_tensor(points, lo, rng, dev):
    p = torch.tensor(np.asarray(points, dtype=np.float32), device=dev)
    return (p - torch.tensor(lo, dtype=torch.float32, device=dev)) / torch.tensor(rng, dtype=torch.float32, device=dev)


# ----------------------------------------------------------------------------------------
# theory encoder
# ----------------------------------------------------------------------------------------
def train_theory(cfg, verbose=True):
    """Train the theory encoder with dropout views.  No parameter perturbation."""
    dev = dev_of()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train_pts = GROUPS["train"]
    lo, rng = minmax_params(ALL_POINTS)
    mu, sd = standardise_params(train_pts, train_pts)
    neg_w, neg_allowed = build_negative_weights(
        cfg["neg"], train_pts, mu, sd, cfg["dmass"], cfg["dlam"], cfg["d0"], dev=dev)

    model = TheoryMLP(cfg["dlat"], cfg["width"], cfg["depth"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4, betas=(0.9, 0.98))

    P = len(train_pts)
    u = minmax_tensor(train_pts, lo, rng, dev)
    point_ids = torch.arange(P, device=dev)

    t0 = time.time()
    hist = []
    for step in range(cfg["steps"]):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, cfg["steps"], max(1, cfg["steps"] // 20), cfg["lr"])
        model.train()
        z_a = model(u)          # dropout mask a
        z_b = model(u)          # dropout mask b
        loss, info = info_nce_two_view(z_a, z_b, cfg["tau"], point_ids, neg_w, neg_allowed)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            hist.append({"step": step, "loss": float(loss.item()), **info})
            if verbose:
                print(f"[theory s{cfg['seed']}] step {step:5d} loss {loss.item():.4f} "
                      f"acc {info['acc']:.3f} nneg {info['n_neg']:.1f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    pts_all = GROUPS["train"] + GROUPS["val"] + GROUPS["test"] + GROUPS["lowstat"]
    with torch.no_grad():
        z_all = model(minmax_tensor(pts_all, lo, rng, dev)).cpu().numpy()
    return {
        "model": model,
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "lo": lo, "rng": rng, "mu": mu, "sd": sd,
        "points": pts_all,
        "z": z_all,
        "hist": hist,
        "cfg": cfg,
    }


# ----------------------------------------------------------------------------------------
# detector encoder
# ----------------------------------------------------------------------------------------
class PhysAug:
    """Azimuthal rotation and eta reflection of an event set.

    Both are symmetries of the detector response, so applying them does not change the
    reconstructed distribution.  They are used only as an optional extra augmentation of
    the detector encoder.
    """

    def __init__(self, data):
        names = [n.lower() for n in getattr(data, "feat_names", [])]
        self.sin_i = [i for i, n in enumerate(names) if n == "sinphi"]
        self.cos_i = [i for i, n in enumerate(names) if n == "cosphi"]
        self.eta_i = [i for i, n in enumerate(names) if n == "eta"]
        self.mean = data.mean
        self.std = data.std

    def __call__(self, x, m):
        B = len(x)
        dev = x.device
        xr = x * self.std.to(dev) + self.mean.to(dev)
        if self.sin_i:
            a = torch.rand(B, 1, device=dev) * 2 * math.pi
            ca, sa = a.cos(), a.sin()
            for si, ci in zip(self.sin_i, self.cos_i):
                s_, c_ = xr[..., si], xr[..., ci]
                xr[..., si], xr[..., ci] = s_ * ca + c_ * sa, c_ * ca - s_ * sa
        if self.eta_i:
            flip = (torch.rand(B, 1, device=dev) < 0.5).float() * 2 - 1
            for ei in self.eta_i:
                xr[..., ei] = xr[..., ei] * flip
        return ((xr - self.mean.to(dev)) / self.std.to(dev)) * m.unsqueeze(-1)


def load_llm_targets(points, cfg):
    """Frozen LLM embeddings for the paired parameter points, PCA-reduced.

    With two paired points the only Qwen-side geometry is the separation
    direction (q_hi - q_lo): PCA degenerates to a single component, and a 1-D
    L2 normalisation has vanishing gradient.  In that case the targets are the
    mean-centred raw card vectors (antipodal unit vectors at full width).
    """
    d = np.load(cfg["llm_npz"], allow_pickle=True)
    qpts = [tuple(int(v) for v in p) for p in d["points"]]
    li = [int(v) for v in d["layer_ids"]].index(int(cfg["llm_layer"]))
    Z = d[f"{cfg['llm_variant']}__{cfg['llm_pooling']}"][:, li, :].astype(np.float64)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    m = {p: i for i, p in enumerate(qpts)}
    Z = Z[[m[p] for p in points]]
    if len(points) <= 2:
        mean = Z.mean(0)
        Qc = Z - mean
        Q = Qc / np.maximum(np.linalg.norm(Qc, axis=1, keepdims=True), 1e-12)

        class _Raw:
            pass

        pca = _Raw()
        pca.mean_ = mean
        pca.components_ = np.zeros((0, Z.shape[1]))
        return Q.astype(np.float32), pca, int(Z.shape[1]), Z.astype(np.float32)
    n_comp = min(int(cfg["cm_dim"]), len(points) - 1)
    pca = PCA(n_components=n_comp, random_state=cfg["seed"]).fit(Z)
    Q = pca.transform(Z)
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    return Q.astype(np.float32), pca, n_comp, Z.astype(np.float32)


def train_detector(cfg, data, verbose=True):
    dev = data.dev
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 7717)

    train_pts = list(cfg.get("train_points") or GROUPS["train"])
    train_split = int(cfg.get("train_split", 0))
    mu, sd = standardise_params(train_pts, train_pts)
    neg_w, neg_allowed = build_negative_weights(
        cfg["neg"], train_pts, mu, sd, cfg["dmass"], cfg["dlam"], cfg["d0"], dev=dev)

    n_summ = data.n_summ if cfg.get("use_summ", True) else 0
    model = DeepSetsDetector(data.n_feat, cfg["dlat"], cfg["width"], cfg["depth"],
                             cfg["pwidth"], cfg["pool"], cfg["dropout"], n_summ=n_summ,
                             backbone=cfg.get("backbone", "deepsets"),
                             tf_d=cfg.get("tf_d", 64), tf_layers=cfg.get("tf_layers", 2),
                             tf_heads=cfg.get("tf_heads", 4)).to(dev)

    cm_w = float(cfg.get("cm_weight", 0.0))
    cm_evt_w = float(cfg.get("cm_evt_weight", 0.0))
    cm_mode = cfg.get("cm_mode", "point")
    cm_head, q_t, q_proto_t, mass_id_t, cm_info = None, None, None, None, None
    if cm_w > 0 or cm_evt_w > 0:
        q_np, pca, n_comp, q_raw = load_llm_targets(train_pts, cfg)
        q_t = torch.tensor(q_np, device=dev)
        q_raw_t = torch.tensor(q_raw, device=dev)
        cm_head = torch.nn.Linear(cfg["dlat"], n_comp, bias=True).to(dev)
        cm_info = {
            "state": {k: v.detach().cpu() for k, v in cm_head.state_dict().items()},
            "mean": pca.mean_.astype(np.float32),
            "components": getattr(pca, "components_", np.zeros((0, q_raw.shape[1]))).astype(np.float32),
            "n_components": int(n_comp),
            "points": np.asarray(train_pts),
            "variant": cfg["llm_variant"], "pooling": cfg["llm_pooling"],
            "layer": int(cfg["llm_layer"]),
            "mode": cm_mode,
        }
        if cm_mode == "mass":
            masses_tr = [p[0] for p in train_pts]
            um = sorted(set(masses_tr))
            mid = np.asarray([um.index(m) for m in masses_tr], dtype=np.int64)
            q_proto = np.stack([q_np[mid == j].mean(0) for j in range(len(um))])
            q_proto = q_proto / np.maximum(np.linalg.norm(q_proto, axis=1, keepdims=True), 1e-12)
            q_proto_t = torch.tensor(q_proto.astype(np.float32), device=dev)
            mass_id_t = torch.tensor(mid, device=dev)
            cm_info["q_proto"] = q_proto.astype(np.float32)
            cm_info["mass_ids"] = mid
            cm_info["masses"] = np.asarray(um)
        if verbose:
            print(f"[prh] cross-modal target: {n_comp} Qwen PCs, "
                  f"{len(train_pts)} paired points, mode={cm_mode}", flush=True)

    params = list(model.parameters())
    if cm_head is not None:
        params += list(cm_head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-4, betas=(0.9, 0.98))

    P, K = len(train_pts), cfg["k"]
    sampler = CursorSampler(data, train_pts, split=train_split, generator=gen)
    point_ids = torch.arange(P, device=dev)
    aug = PhysAug(data) if cfg.get("phys_aug") else None

    t0 = time.time()
    hist = []
    for step in range(cfg["steps"]):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, cfg["steps"], max(1, cfg["steps"] // 20), cfg["lr"])
        idx = sampler.block(2 * K)                    # (P, 2K), disjoint within a step
        idx_a = idx[:, :K].reshape(-1)
        idx_b = idx[:, K:].reshape(-1)
        xa, ma = data.get(idx_a)
        xb, mb = data.get(idx_b)
        sa = data.get_summ(idx_a)
        sb = data.get_summ(idx_b)
        if aug is not None:
            xa = aug(xa, ma)
            xb = aug(xb, mb)
        ew = cfg.get("evt_weight", 0.0)
        sw = cfg.get("sup_weight", 0.0)
        if ew > 0 or sw > 0 or cm_evt_w > 0:
            z_a, e_a = model(xa, ma, P, K, sa, return_events=True)
            z_b, e_b = model(xb, mb, P, K, sb, return_events=True)
        else:
            z_a = model(xa, ma, P, K, sa)
            z_b = model(xb, mb, P, K, sb)
        loss, info = info_nce_two_view(z_a, z_b, cfg["tau"], point_ids, neg_w, neg_allowed)
        if ew > 0:
            ge_a = F.normalize(model.g(e_a), dim=-1)
            ge_b = F.normalize(model.g(e_b), dim=-1)
            loss_evt, acc_evt = event_prototype_loss(
                ge_a, ge_b, P, K, cfg.get("evt_tau", 0.1))
            loss = loss + ew * loss_evt
            info["acc_evt"] = acc_evt
        if sw > 0:
            ge_a = F.normalize(model.g(e_a), dim=-1)
            ge_b = F.normalize(model.g(e_b), dim=-1)
            lab = point_ids.repeat_interleave(K)
            loss_sup, acc_sup = supcon_loss(
                torch.cat([ge_a, ge_b], dim=0), torch.cat([lab, lab], dim=0),
                cfg["sup_tau"])
            loss = loss + sw * loss_sup
            info["loss_sup"] = float(loss_sup.item())
            info["acc_sup"] = acc_sup
        if cm_w > 0:
            h_a = F.normalize(cm_head(z_a), dim=-1)
            h_b = F.normalize(cm_head(z_b), dim=-1)
            if cm_mode == "mass":
                lc_a, acc_a = mass_prototype_loss(h_a, q_proto_t, mass_id_t, cfg["cm_tau"])
                lc_b, acc_b = mass_prototype_loss(h_b, q_proto_t, mass_id_t, cfg["cm_tau"])
            else:
                lc_a, ia = info_nce_two_view(h_a, q_t, cfg["cm_tau"], point_ids)
                lc_b, ib = info_nce_two_view(h_b, q_t, cfg["cm_tau"], point_ids)
                acc_a, acc_b = ia["acc"], ib["acc"]
            loss_cm = 0.5 * (lc_a + lc_b)
            loss = loss + cm_w * loss_cm
            info["loss_cm"] = float(loss_cm.item())
            info["acc_cm"] = 0.5 * (acc_a + acc_b)
        if cm_evt_w > 0:
            ge_a = F.normalize(model.g(e_a), dim=-1)
            ge_b = F.normalize(model.g(e_b), dim=-1)
            he_a = F.normalize(cm_head(ge_a), dim=-1)
            he_b = F.normalize(cm_head(ge_b), dim=-1)
            lab = point_ids.repeat_interleave(K)
            le_a, ae_a = mass_prototype_loss(he_a, q_t, lab, cfg["cm_tau"])
            le_b, ae_b = mass_prototype_loss(he_b, q_t, lab, cfg["cm_tau"])
            loss_evt_cm = 0.5 * (le_a + le_b)
            loss = loss + cm_evt_w * loss_evt_cm
            info["loss_evt_cm"] = float(loss_evt_cm.item())
            info["acc_evt_cm"] = 0.5 * (ae_a + ae_b)
            cev_al = float(cfg.get("cm_evt_align_weight", 0.0))
            if cev_al > 0:
                # direct per-event cosine regression onto the raw card rep
                al_a = (1.0 - (he_a * q_raw_t[lab]).sum(-1)).mean()
                al_b = (1.0 - (he_b * q_raw_t[lab]).sum(-1)).mean()
                loss_align = 0.5 * (al_a + al_b)
                loss = loss + cev_al * loss_align
                info["loss_evt_align"] = float(loss_align.item())
                info["cos_evt_align"] = float(1.0 - loss_align)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            hist.append({"step": step, "loss": float(loss.item()), **info})
            if verbose:
                extra = ""
                if "acc_sup" in info:
                    extra += f" sup {info['acc_sup']:.3f}"
                if "acc_cm" in info:
                    extra += f" acc_cm {info['acc_cm']:.3f} cm {info.get('loss_cm', 0):.3f}"
                if "acc_evt_cm" in info:
                    extra += (f" acc_evt_cm {info['acc_evt_cm']:.3f} "
                              f"evtcm {info.get('loss_evt_cm', 0):.3f}")
                if "cos_evt_align" in info:
                    extra += f" align {info['cos_evt_align']:.3f}"
                print(f"[detector s{cfg['seed']}] step {step:5d} loss {loss.item():.4f} "
                      f"acc {info['acc']:.3f} nneg {info['n_neg']:.1f}{extra} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    if cm_head is not None:
        cm_info["state"] = {k: v.detach().cpu() for k, v in cm_head.state_dict().items()}
    return {
        "model": model,
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": cfg,
        "hist": hist,
        "cm": cm_info,
    }


@torch.no_grad()
def embed_detector(model, data, points, split=2, k=64, n_rep=8, seed=0, chunk=4096,
                   min_per_rep=32):
    """Representation of each parameter point: mean of the unit embeddings of n_rep
    independent event sets, renormalised.  Returns (z, within-point scatter).

    The set size is reduced automatically for parameter points whose evaluation pool is
    too small to supply ``n_rep`` disjoint sets of ``k`` events.
    """
    dev = data.dev
    gen = torch.Generator().manual_seed(seed)
    zs, scat = [], []
    for p in points:
        pool = len(data.pool(p, split))
        kk, nr = k, n_rep
        while kk > min_per_rep and kk * nr > pool:
            if nr > 1:
                nr -= 1
            else:
                kk = max(min_per_rep, pool)
                break
        ev = sample_disjoint(data, p, nr * kk, split, generator=gen)
        outs = []
        for s in range(0, nr * kk, chunk):
            sl = ev[s:s + chunk]
            x, m = data.get(sl)
            su = data.get_summ(sl)
            # events are laid out view-major: view v holds events [v*kk, (v+1)*kk)
            nv = len(sl) // kk
            outs.append(model(x, m, nv, kk, su))
        u = torch.cat(outs, 0)                          # (nr, d)
        z = u.mean(0)
        z = z / z.norm().clamp_min(1e-12)
        zs.append(z.cpu().numpy())
        scat.append(float(u.std(0).mean().item()) if u.shape[0] > 1 else 0.0)
    return np.asarray(zs), np.asarray(scat)


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
def save_run(tag, outdir, payload):
    d = os.path.join(outdir, tag)
    os.makedirs(d, exist_ok=True)
    torch.save({k: payload[k] for k in ("state", "cfg")}, os.path.join(d, "state.pt"))
    if payload.get("cm") is not None:
        torch.save(payload["cm"], os.path.join(d, "cm.pt"))
    np.savez(os.path.join(d, "emb.npz"), z=payload["z"], points=np.asarray(payload["points"]),
             scatter=payload.get("scatter", np.zeros(0)))
    with open(os.path.join(d, "hist.json"), "w") as f:
        json.dump({"hist": payload["hist"], "cfg": payload["cfg"]}, f, indent=2, default=str)
    print(f"[prh] saved {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["theory", "detector"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--neg", default="standard")
    ap.add_argument("--dmass", type=float, default=50.0)
    ap.add_argument("--dlam", type=float, default=1.0)
    ap.add_argument("--d0", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dlat", type=int, default=16)
    ap.add_argument("--phys-aug", action="store_true")
    ap.add_argument("--evt-weight", type=float, default=0.0)
    ap.add_argument("--evt-tau", type=float, default=0.1)
    ap.add_argument("--sup-weight", type=float, default=0.0)
    ap.add_argument("--sup-tau", type=float, default=0.1)
    ap.add_argument("--holdout-masses", default="")
    ap.add_argument("--lambda", dest="lam", type=int, default=None)
    ap.add_argument("--cm-weight", type=float, default=0.0)
    ap.add_argument("--cm-mode", default="point", choices=["point", "mass"])
    ap.add_argument("--cm-tau", type=float, default=0.1)
    ap.add_argument("--cm-dim", type=int, default=16)
    ap.add_argument("--llm-npz",
                    default="fisher_geometry/runs/prh_llm_qwen3_8b/llm_embeddings.npz")
    ap.add_argument("--llm-layer", type=int, default=18)
    ap.add_argument("--backbone", default="deepsets", choices=["deepsets", "transformer"])
    ap.add_argument("--no-summ", action="store_true")
    ap.add_argument("--cache", default="fisher_geometry/cache/prh_cache_full.pt")
    ap.add_argument("--out", default="fisher_geometry/runs/prh_platonic")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    dev = dev_of()
    if args.mode == "theory":
        cfg = theory_cfg(seed=args.seed, steps=args.steps, neg=args.neg,
                         dmass=args.dmass, dlam=args.dlam, d0=args.d0,
                         tau=args.tau, dlat=args.dlat, out=args.out)
        if args.dropout is not None:
            cfg["dropout"] = args.dropout
        if args.lr is not None:
            cfg["lr"] = args.lr
        res = train_theory(cfg)
        tag = args.tag or f"theory_s{args.seed}_{args.neg}_d{args.dropout}"
        save_run(tag, os.path.join(args.out, "encoders"), res)
    else:
        if "full" not in args.cache:
            build_cache("fisher_geometry/cache.pt", args.cache)
        data = PrhData(args.cache, dev)
        holdout = [int(x) for x in args.holdout_masses.split(",") if x.strip()]
        all_pts = (GROUPS["train"] + GROUPS["val"] + GROUPS["test"]
                   + GROUPS["lowstat"])
        train_points = list(all_pts)
        if holdout:
            train_points = [p for p in train_points if p[0] not in holdout]
        if args.lam is not None:
            train_points = [p for p in train_points if p[1] == args.lam]
        cfg = detector_cfg(seed=args.seed, steps=args.steps, neg=args.neg,
                           dmass=args.dmass, dlam=args.dlam, d0=args.d0,
                           k=args.k, tau=args.tau, dlat=args.dlat,
                           backbone=args.backbone, use_summ=not args.no_summ,
                            phys_aug=args.phys_aug, evt_weight=args.evt_weight,
                            evt_tau=args.evt_tau, sup_weight=args.sup_weight,
                            sup_tau=args.sup_tau, train_points=train_points,
                           cm_weight=args.cm_weight, cm_mode=args.cm_mode,
                           cm_tau=args.cm_tau, cm_dim=args.cm_dim, llm_npz=args.llm_npz,
                           llm_layer=args.llm_layer, out=args.out)
        if args.lr is not None:
            cfg["lr"] = args.lr
        res = train_detector(cfg, data)
        pts = GROUPS["train"] + GROUPS["val"] + GROUPS["test"] + GROUPS["lowstat"]
        z, scat = embed_detector(res["model"], data, pts, split=2, k=args.k, n_rep=8,
                                  seed=args.seed + 13)
        res["z"] = z
        res["points"] = pts
        res["scatter"] = scat
        tag = args.tag or f"detector_s{args.seed}_{args.neg}_k{args.k}"
        save_run(tag, os.path.join(args.out, "encoders"), res)


if __name__ == "__main__":
    main()
