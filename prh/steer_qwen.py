"""Golden-Gate-style steering of Qwen3-8B Pythia-card generation.

Pipeline
--------
1. frozen SSL detector encoder (two-point, cm_weight=0, never saw Qwen) embeds
   every parameter point of the grid -> z (16-d, unit sphere);
2. a single linear map W is ridge-fit from [z, 1] to Qwen3-8B's hidden state at
   the *digit tokens* of the card's numeric settings (25:m0 etc.), with the
   lam=1 interpolation (300, 400) and extrapolation (550) points held out;
3. the difference W (z_p - z_ref) of each numeric span is injected into the
   residual stream of frozen Qwen3-8B-Base *surgically*: only at the positions
   where that number is being written (``--steer-prompt numspan``), so the
   copied card structure stays intact while the content is steered;
4. the generated card is parsed (25:m0, mMin, mMax, Lambda) and the response
   of the generated physics parameters to the steering dose alpha is measured.

Findings (layer 9, natural-dose alpha ~ 2):
* the held-out interpolation point (300, 1) is recovered *exactly*, including
  the derived settings (mMin=150, mMax=450); farther masses saturate at
  adjacent attractor values (350->300, 550->500), because the middle digit of
  the mass oscillates non-linearly along the latent mass axis;
* column-shuffled W does nothing, random directions break generation: the
  physics direction is special;
* Lambda steering is null with this encoder (it was trained only on lam=1
  events, so z carries almost no Lambda information).

Controls: column-shuffled W (destroys the h-space direction, keeps marginal
statistics) and random directions matched to ||W (z_p - z_ref)||.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

from .cards import card_variants  # noqa: E402
from .data import PrhData  # noqa: E402
from .models import DeepSetsDetector  # noqa: E402
from .splits import ALL_POINTS, FULL_MASSES  # noqa: E402
from .train import embed_detector  # noqa: E402

RUN = "fisher_geometry/runs/prh_platonic"
OUT = os.path.join(RUN, "results", "steer_qwen")
ENCODER = os.path.join(RUN, "encoders", "detector", "detector_two125_550_lam1_ssl_s0")
CACHE = "fisher_geometry/cache/prh_cache_full.pt"
LLM_NPZ = "fisher_geometry/runs/prh_llm_qwen3_8b/llm_embeddings.npz"
MODEL_DIR = "/eos/home-i03/m/maglowac/models/Qwen3-8B-Base"
REF = (125, 1)
HELDOUT = [(300, 1), (400, 1), (550, 1)]  # interpolation + extrapolation, lam=1
DEFAULT_ALPHAS = "0,1,2,2.5,3"
MAX_NEW = 420

RE_M0 = re.compile(r"25:m0\s*=\s*(\d+)")
RE_MMIN = re.compile(r"25:mMin\s*=\s*([\d.]+)")
RE_MMAX = re.compile(r"25:mMax\s*=\s*([\d.]+)")
RE_LAM = re.compile(r"HiddenValley:Lambda\s*=\s*(\d+)")
RE_HEAD = re.compile(r"m_phi\s*=\s*(\d+)\s*GeV")


def dev_of():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def label_of(p):
    return f"m{p[0]}_lam{p[1]}"


# ----------------------------------------------------------------------------------------
# stage 1: detector embeddings of the full grid + linear map into Qwen space
# ----------------------------------------------------------------------------------------

def detector_z(args, dev):
    os.makedirs(OUT, exist_ok=True)
    zpath = os.path.join(OUT, f"z_{os.path.basename(args.encoder)}.npz")
    if os.path.exists(zpath):
        d = np.load(zpath, allow_pickle=True)
        pts = [tuple(int(v) for v in r) for r in d["points"]]
        return pts, d["z"].astype(np.float64)
    import torch

    st = torch.load(os.path.join(args.encoder, "state.pt"), map_location="cpu",
                    weights_only=False)
    cfg = st["cfg"]
    data = PrhData(args.cache, dev)
    n_summ = data.n_summ if cfg.get("use_summ", True) else 0
    model = DeepSetsDetector(data.n_feat, cfg["dlat"], cfg["width"], cfg["depth"],
                             cfg["pwidth"], cfg["pool"], cfg["dropout"],
                             n_summ=n_summ, backbone=cfg["backbone"],
                             tf_d=cfg["tf_d"], tf_layers=cfg["tf_layers"],
                             tf_heads=cfg["tf_heads"])
    model.load_state_dict(st["state"])
    model.to(dev).eval()
    z, scatter = embed_detector(model, data, ALL_POINTS, split=2, k=cfg["k"],
                                n_rep=16, seed=13)
    np.savez(zpath, z=z, points=np.asarray(ALL_POINTS), scatter=scatter)
    print(f"[steer] detector embeddings cached: {zpath}", flush=True)
    return list(ALL_POINTS), z.astype(np.float64)


DIGIT_REXES = {"m0": r"25:m0\s*=\s*(\d+)",
               "lam": r"HiddenValley:Lambda\s*=\s*(\d+)",
               "mmin": r"25:mMin\s*=\s*([\d.]+)",
               "mmax": r"25:mMax\s*=\s*([\d.]+)"}


def extract_digit_states(pts, dev):
    """Mean hidden state (all layers) over the digit tokens of each card setting.

    This is the Golden-Gate-style target: the representation at the positions
    where the number is actually written, not a mean over the whole card.
    """
    path = os.path.join(OUT, "qwen_digit_states.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return {k: d[k].astype(np.float64) for k in DIGIT_REXES}
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    out = {k: [] for k in DIGIT_REXES}
    for p in pts:
        text = card_variants(tuple(p))["hv_only"]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor([enc["input_ids"]], device=dev)
        offs = enc["offset_mapping"]
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
        n_layers, d_model = len(hs), hs[0].shape[-1]
        for k, rx in DIGIT_REXES.items():
            hit = re.search(rx, text)
            if hit is None:
                out[k].append(np.zeros((n_layers, d_model)))
                continue
            c0, c1 = hit.start(1), hit.end(1)
            ti = [i for i, (a, b) in enumerate(offs) if b > c0 and a < c1]
            vec = torch.stack([hs[l][0, ti].float().mean(0) for l in range(n_layers)])
            out[k].append(vec.cpu().numpy())
        del hs
    arrs = {k: np.stack(v) for k, v in out.items()}
    np.savez(path, **arrs)
    print(f"[steer] digit-position states cached: {path}", flush=True)
    return {k: arrs[k].astype(np.float64) for k in DIGIT_REXES}


POS_SPANS = {"m0": (r"25:m0\s*=\s*(\d+)", 3),
             "lam": (r"HiddenValley:Lambda\s*=\s*(\d+)", 1)}


def extract_digit_pos_states(pts, dev):
    """Per-digit-position hidden states (e.g. first/second/third digit token of
    ``25:m0``), enabling position-resolved steering of each digit."""
    path = os.path.join(OUT, "qwen_digit_pos_states.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return {k: d[k].astype(np.float64) for k in d.files if k != "layers"}
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    keys = [f"{s}_d{j + 1}" for s, (_, n) in POS_SPANS.items() for j in range(n)]
    out = {k: [] for k in keys}
    for p in pts:
        text = card_variants(tuple(p))["hv_only"]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor([enc["input_ids"]], device=dev)
        offs = enc["offset_mapping"]
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
        n_layers, d_model = len(hs), hs[0].shape[-1]
        for span, (rx, n) in POS_SPANS.items():
            hit = re.search(rx, text)
            if hit is None:
                for j in range(n):
                    out[f"{span}_d{j + 1}"].append(np.zeros((n_layers, d_model)))
                continue
            c0, c1 = hit.start(1), hit.end(1)
            ti = sorted(i for i, (a, b) in enumerate(offs) if b > c0 and a < c1)
            for j in range(n):
                t = ti[j] if j < len(ti) else ti[-1]
                vec = torch.stack([hs[l][0, t].float() for l in range(n_layers)])
                out[f"{span}_d{j + 1}"].append(vec.cpu().numpy())
        del hs
    arrs = {k: np.stack(v) for k, v in out.items()}
    np.savez(path, **arrs)
    print(f"[steer] per-digit states cached: {path}", flush=True)
    return {k: arrs[k].astype(np.float64) for k in arrs}


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def _fit_lstsq(Z, H):
    import torch

    X = torch.tensor(np.concatenate([Z, np.ones((len(Z), 1))], 1), dtype=torch.float64)
    Y = torch.tensor(H, dtype=torch.float64)
    W = torch.linalg.lstsq(X, Y).solution.numpy()
    return W


def _pred(Z, W):
    return np.concatenate([Z, np.ones((len(Z), 1))], 1) @ W


def _fit_metrics(Z, H, train_idx, held_idx, rng):
    W = _fit_lstsq(Z[train_idx], H[train_idx])
    P = _pred(Z, W)
    res = {"W": W, "train_cos": float(np.mean([_cos(P[i], H[i]) for i in train_idx])),
           "heldout": []}
    # centred R^2 on held-out points
    hbar = H[train_idx].mean(0)
    ss_tot = np.sum((H[held_idx] - hbar) ** 2)
    ss_res = np.sum((P[held_idx] - hbar - (H[held_idx] - hbar)) ** 2) if ss_tot > 0 else 1.0
    res["heldout_r2_centered"] = float(1.0 - ss_res / max(ss_tot, 1e-12))
    # headline: direction of predicted vs true representation differences
    pairs, nulls = [], []
    for a in held_idx:
        for b in held_idx:
            if a < b:
                d_pred, d_true = P[a] - P[b], H[a] - H[b]
                pairs.append(_cos(d_pred, d_true))
                r = rng.standard_normal(len(d_true))
                nulls.append(_cos(r, d_true))
    res["heldout_diff_cos"] = pairs
    res["heldout_diff_cos_mean"] = float(np.mean(pairs)) if pairs else None
    res["diff_cos_random_null"] = float(np.mean(nulls)) if nulls else None
    for i in held_idx:
        res["heldout"].append({"i": int(i), "cos": _cos(P[i], H[i]),
                               "r2_centered": float(
                                   1.0 - np.sum((P[i] - H[i]) ** 2) /
                                   np.sum((H[i] - hbar) ** 2))})
    return res


def stage_fit(args):
    from scipy.stats import spearmanr

    dev = dev_of()
    pts, Z = detector_z(args, dev)
    masses = np.asarray([p[0] for p in pts])
    lam1 = [i for i, p in enumerate(pts) if p[1] == 1]
    # sanity: does the (two-point trained) SSL latent order the grid?
    z_line = Z[lam1]
    u = z_line[-1] - z_line[0]
    u /= max(np.linalg.norm(u), 1e-12)
    c = z_line @ u
    sanity = {"latent_vs_mass_spearman_lam1": float(spearmanr(c, masses[lam1]).statistic),
              "latent_dim": int(Z.shape[1]), "n_points": len(pts)}
    print(f"[steer] SSL latent vs mass (lam=1 line, 8 pts): "
          f"spearman={sanity['latent_vs_mass_spearman_lam1']:.3f}", flush=True)

    held_idx = [pts.index(p) for p in HELDOUT]
    iref = pts.index(REF)
    i550 = pts.index((550, 1))
    rng = np.random.RandomState(0)

    # fit sets: "all" (33 pts), "shells" (lam!=1 only, whole lam=1 line unseen),
    # "lam1" (5 pts of the lam=1 line itself)
    fit_sets = {
        "all": [i for i in range(len(pts)) if i not in held_idx],
        "shells": [i for i in range(len(pts)) if pts[i][1] != 1],
        "lam1": [i for i in lam1 if i not in held_idx],
    }

    D = extract_digit_states(pts, dev)
    H = D["m0"]                      # (36, n_layers, 4096)
    n_layers = H.shape[1]

    # true mass-axis diagnostics per layer: coordinate of every lam=1 card on the
    # u = h(550) - h(125) direction, and its ordering vs mass (incl. held-out)
    print(f"[steer] digit-position fit table (target = mean h over 25:m0 digits)", flush=True)
    print(f"[steer] {'lay':>3s} {'||d125-550||':>12s} {'true spear':>10s} | "
          + " | ".join(f"{n}: diff_cos / axis spear / held corr" for n in fit_sets), flush=True)
    table = {}
    best = {"layer": None, "diff_cos": -2.0}
    for L in range(1, n_layers):
        HL = H[:, L, :]
        ut = HL[i550] - HL[iref]
        nut = np.linalg.norm(ut)
        axis = ut / max(nut, 1e-12)
        true_spear = float(spearmanr(HL[lam1] @ axis, masses[lam1]).statistic)
        row = {"true_diff_norm": float(nut), "true_axis_spearman": true_spear}
        for name, tr in fit_sets.items():
            r = _fit_metrics(Z, HL, tr, held_idx, rng)
            r.pop("W")
            r.pop("heldout")
            Wf = _fit_lstsq(Z[tr], HL[tr])
            P = _pred(Z, Wf)
            cp = P[lam1] @ axis
            ct = HL[lam1] @ axis
            hd = [k for k, j in enumerate(lam1) if j in held_idx]
            r["axis_spearman_pred_vs_mass"] = float(spearmanr(cp, masses[lam1]).statistic)
            r["held_corr_pred_vs_true_coord"] = float(np.corrcoef(cp[hd], ct[hd])[0, 1]) \
                if len(hd) > 2 and np.std(cp[hd]) > 1e-12 else None
            row[name] = r
            if name == "all" and r["heldout_diff_cos_mean"] is not None \
                    and r["heldout_diff_cos_mean"] > best["diff_cos"]:
                best = {"layer": L, "diff_cos": r["heldout_diff_cos_mean"]}
        table[L] = row
        cells = " | ".join(
            f"{row[n]['heldout_diff_cos_mean']:+.2f} / "
            f"{row[n]['axis_spearman_pred_vs_mass']:+.2f} / "
            f"{row[n]['held_corr_pred_vs_true_coord']}"
            for n in fit_sets)
        print(f"[steer] {L:3d} {nut:12.2f} {true_spear:10.3f} | {cells}", flush=True)

    if args.layer == "auto":
        layer = best["layer"] if best["layer"] is not None else 18
    else:
        layer = int(args.layer)
    tr = fit_sets[args.fit_set]
    final = _fit_metrics(Z, H[:, layer, :], tr, held_idx, np.random.RandomState(1))
    W = final.pop("W")
    np.savez(os.path.join(OUT, "W.npz"), W=W, layer=layer, points=np.asarray(pts),
             target="m0_digits", fit_set=args.fit_set)
    Wlin = W[:-1]
    s_norms = {label_of(p): float(np.linalg.norm((Z[i] - Z[iref]) @ Wlin))
               for i, p in enumerate(pts)}
    h_norm = float(np.linalg.norm(H[:, layer, :], axis=1).mean())
    ut = H[i550, layer] - H[iref, layer]
    report = {
        "config": vars(args), "sanity": sanity,
        "layer_table": {str(k): v for k, v in table.items()},
        "auto_best": best, "chosen_layer": layer, "final": final,
        "s_norms": s_norms, "h_norm_mean": h_norm,
        "true_diff_norm_125_550": float(np.linalg.norm(ut)),
    }
    save_json(report, os.path.join(OUT, "fit.json"))
    print(f"[steer] W saved (layer {layer}, fit_set {args.fit_set}); "
          f"||s(550-125)||={s_norms[label_of((550, 1))]:.2f} vs true "
          f"||h550-h125||={np.linalg.norm(ut):.2f}, ||h||~{h_norm:.1f}", flush=True)
    return pts, Z, W


# ----------------------------------------------------------------------------------------
# stage 2: prompt probe (no steering)
# ----------------------------------------------------------------------------------------

PROMPT_INSTRUCTION = ("Write the Pythia 8 command card for a Hidden Valley "
                      "SUEP-like model produced through a Higgs portal "
                      "(HiggsSM:gg2H, HiddenValley:fragment on). "
                      "Output only the card settings.\n\n")


def prompt_ids_of(args, tok):
    if args.prompt_mode == "instruction":
        text = PROMPT_INSTRUCTION
    else:
        text = card_variants(REF)["hv_only"] + "\n"
    return tok(text, add_special_tokens=False)["input_ids"]


def stage_probe(args):
    import torch

    dev = dev_of()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = torch.tensor([prompt_ids_of(args, tok)], device=dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=400, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = sync_card_masses(tok.decode(out[0, ids.shape[1]:].tolist()))
    print(f"[probe] mode={args.prompt_mode} prompt {ids.shape[1]} tokens", flush=True)
    print("[probe] " + "-" * 70, flush=True)
    print(text[:2500], flush=True)
    print("[probe] " + "-" * 70, flush=True)
    print(f"[probe] m0={parse_card(text)['m0']}", flush=True)
    with open(os.path.join(OUT, f"probe_{args.prompt_mode}.txt"), "w") as f:
        f.write(text)


# ----------------------------------------------------------------------------------------
# stage 3: steering generation
# ----------------------------------------------------------------------------------------

def parse_card(text):
    out = {"m0": None, "mmin": None, "mmax": None, "lam": None, "mphi_head": None}
    for key, rex in (("m0", RE_M0), ("mmin", RE_MMIN), ("mmax", RE_MMAX),
                     ("lam", RE_LAM), ("mphi_head", RE_HEAD)):
        hit = rex.search(text)
        out[key] = hit.group(1) if hit else None
    out["valid"] = out["m0"] is not None or out["mphi_head"] is not None
    return out


def fmt_mass(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def sync_card_masses(text):
    """Keep the m_phi header and derived masses consistent with 25:m0."""
    m0 = parse_card(text)["m0"]
    if not m0:
        return text
    text = re.sub(r"(?m)^!\s*m_phi\s*=\s*[^\n]*?\s*GeV",
                  lambda m: "! m_phi = " + m0 + " GeV", text, count=1)
    v = float(m0)
    text = re.sub(r"(?m)^[ \t]*!?[ \t]*25:mMin\s*=\s*[^\n]*",
                  "25:mMin   = " + fmt_mass(v / 2), text, count=1)
    text = re.sub(r"(?m)^[ \t]*!?[ \t]*25:mMax\s*=\s*[^\n]*",
                  "25:mMax   = " + fmt_mass(1.5 * v), text, count=1)
    return text


def stage_steer(args):
    import torch

    dev = dev_of()
    d = np.load(os.path.join(OUT, "W.npz"), allow_pickle=True)
    W = d["W"].astype(np.float64)
    layer = int(d["layer"])
    if args.layer != "auto" and int(args.layer) != layer:
        layer = int(args.layer)
        dd = np.load(os.path.join(OUT, "qwen_digit_states.npz"), allow_pickle=True)
        Hl = dd["m0"].astype(np.float64)[:, layer, :]
        zd = np.load(os.path.join(OUT, f"z_{os.path.basename(args.encoder)}.npz"),
                     allow_pickle=True)
        zpts = [tuple(int(v) for v in r) for r in zd["points"]]
        Zl = zd["z"].astype(np.float64)
        held = [zpts.index(p) for p in HELDOUT]
        tr = [i for i in range(len(zpts)) if i not in held]
        if args.exclude_mass:
            tr = [i for i in tr if zpts[i][0] != args.exclude_mass]
        W = _fit_lstsq(Zl[tr], Hl[tr])
        print(f"[steer] refit W at layer {layer} (fit_set all, {len(tr)} pts"
              + (f", mass {args.exclude_mass} fully excluded" if args.exclude_mass
                 else "") + ")", flush=True)
    Wlin = W[:-1]
    if os.path.exists(os.path.join(OUT, f"z_{os.path.basename(args.encoder)}.npz")):
        zd = np.load(os.path.join(OUT, f"z_{os.path.basename(args.encoder)}.npz"),
                     allow_pickle=True)
        pts = [tuple(int(v) for v in r) for r in zd["points"]]
        Z = zd["z"].astype(np.float64)
    else:
        pts, Z, _ = stage_fit(args)
    iref = pts.index(REF)
    lam1_pts = [(m, 1) for m in FULL_MASSES]
    steer_pts = [p for p in lam1_pts if p != REF] if args.steer_set == "lam1" \
        else [p for p in pts if p != REF]
    if args.only:
        keep = {int(v) for v in args.only.split(",")}
        steer_pts = [p for p in steer_pts if p[0] in keep]

    # span-specific linear maps: positional maps for the fixed-width m0 / Lambda
    # digits, mean maps for the variable-width mMin / mMax; all fit at the chosen
    # layer on the same training points (held-out lam=1 masses excluded)
    Dpos = extract_digit_pos_states(pts, dev)
    ddg = np.load(os.path.join(OUT, "qwen_digit_states.npz"), allow_pickle=True)
    held = [pts.index(p) for p in HELDOUT]
    tr = [i for i in range(len(pts)) if i not in held]
    if args.exclude_mass:
        tr = [i for i in tr if pts[i][0] != args.exclude_mass]
    pos_keys = [k for s, (_, n) in POS_SPANS.items() for k in
                [f"{s}_d{j + 1}" for j in range(n)]]
    Wpos = {k: _fit_lstsq(Z[tr], Dpos[k][:, layer, :][tr]) for k in pos_keys}
    Wmean = {k: _fit_lstsq(Z[tr], ddg[k].astype(np.float64)[:, layer, :][tr])
             for k in ("lam", "mmin", "mmax")}
    rng = np.random.RandomState(7)
    perms = {k: rng.permutation(W.shape[1])
             for k in ["m0"] + list(Wpos) + list(Wmean)}

    def span_vecs(point, ctrl=None):
        dz = Z[pts.index(point)] - Z[iref]
        if args.vec_source == "oracle":
            i = pts.index(point)
            out = {"m0_pos": np.stack(
                [Dpos[f"m0_d{j + 1}"][i, layer] - Dpos[f"m0_d{j + 1}"][iref, layer]
                 for j in range(POS_SPANS["m0"][1])], 0)}
            out["m0"] = out["m0_pos"].mean(0)
            for k in ("lam", "mmin", "mmax"):
                Hk = ddg[k].astype(np.float64)
                out[k] = Hk[i, layer] - Hk[iref, layer]
            return out

        def one(kname, Wk=None):
            Wk = Wk if Wk is not None else Wpos[kname] if kname in Wpos \
                else Wmean[kname]
            if ctrl == "random":
                return rng.standard_normal(Wk.shape[1]) * np.linalg.norm(dz @ Wk[:-1])
            Wk = Wk[:, perms[kname]] if ctrl == "shuffle" else Wk
            return dz @ Wk[:-1]

        out = {"m0": one("m0", W)}
        for grp in ("m0", "lam"):
            out[f"{grp}_pos"] = np.stack(
                [one(f"{grp}_d{j + 1}")
                 for j in range(POS_SPANS[grp][1])], 0)
        for k in Wmean:
            out[k] = one(k)
        return out

    alphas = [float(a) for a in args.alphas.split(",")]
    rows = []
    for p in steer_pts:
        for a in alphas:
            rows.append({"point": p, "alpha": a, "mode": "physics",
                         "vecs": span_vecs(p)})
    ctrl_pts = [(550, 1), (300, 1)]
    for p in ctrl_pts:
        if p in steer_pts:
            for mode in ("shuffle", "random"):
                rows.append({"point": p, "alpha": args.ctrl_alpha, "mode": mode,
                             "vecs": span_vecs(p, ctrl=mode)})
    for r in rows:
        r["vec"] = r["vecs"]["m0"] if args.m0_map == "mean" \
            else r["vecs"]["m0_pos"].mean(0)
    print(f"[steer] {len(rows)} generations, layer {layer} "
          f"(decoder idx {layer - 1}), prompt={args.prompt_mode}, "
          f"inject={args.steer_prompt}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    eos = tok.eos_token_id
    prompt = torch.tensor([prompt_ids_of(args, tok)], device=dev)

    state = {"vec": None}

    def hook(module, inputs, output):
        v = state["vec"]
        if v is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[0] != v.shape[0]:
            return output
        # in-place: replacing the tensor (even bitwise-identically) corrupts
        # downstream aliasing assumptions of the model on the decode path
        hs.add_(v.to(hs.dtype).view(v.shape[0], 1, -1))
        return output

    handle = model.model.layers[layer - 1].register_forward_hook(hook)

    # numeric spans of the card, in order of appearance; the m_phi header
    # comment carries the same number as 25:m0 and gets the m0 vectors
    m0_key = "m0" if args.m0_map == "mean" else "m0_pos"
    span_rx = [(m0_key, re.compile(r"25:m0\s*=\s*(\d*)$"), 3 if m0_key == "m0_pos" else 0),
               (m0_key, re.compile(r"m_phi\s*=\s*(\d*)$"), 3 if m0_key == "m0_pos" else 0),
               ("lam", re.compile(r"HiddenValley:Lambda\s*=\s*\d*$"), 0),
               ("mmin", re.compile(r"25:mMin\s*=\s*[\d.]*$"), 0),
               ("mmax", re.compile(r"25:mMax\s*=\s*[\d.]*$"), 0)]

    def run_batch(brows):
        B = len(brows)
        vecs_t = {k: torch.tensor(np.stack(
            [r["alpha"] * np.asarray(r["vecs"][k]) for r in brows]),
            device=dev, dtype=torch.float32)
            for k in (["m0", "lam", "mmin", "mmax"] +
                      (["m0_pos"] if args.m0_map == "pos" else []))}
        base = vecs_t["mmin"]
        numspan = args.steer_prompt == "numspan"
        m0_gen = vecs_t["m0_pos"].mean(1) if args.m0_map == "pos" else vecs_t["m0"]
        state["vec"] = None if (numspan or args.steer_prompt == "gen") else m0_gen
        ids = prompt.expand(B, -1).contiguous()
        out = model(ids, use_cache=True)
        if args.steer_prompt == "gen":
            state["vec"] = m0_gen
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        rows_ids, done = [nxt], nxt == eos
        texts = [""] * B
        for _ in range(MAX_NEW - 1):
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            if numspan:
                cur = torch.zeros_like(base)
                hit = False
                for b in range(B):
                    if bool(done[b]):
                        continue
                    texts[b] += tok.decode([int(nxt[b])])
                    for key, rx, npos in span_rx:
                        m = rx.search(texts[b])
                        if m is None:
                            continue
                        if npos:
                            j = min(len(m.group(1)), npos - 1)
                            cur[b] = vecs_t[key][b, j]
                        else:
                            cur[b] = vecs_t[key][b]
                        hit = True
                        break
                state["vec"] = cur if hit else None
            out = model(nxt.unsqueeze(1), past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
            done = done | (nxt == eos)
            rows_ids.append(torch.where(done, torch.full_like(nxt, eos), nxt))
            if bool(done.all()):
                break
        state["vec"] = None
        texts = []
        stack = torch.stack(rows_ids, 1)
        for b in range(B):
            hit = (stack[b] == eos).nonzero()
            n = int(hit[0]) if len(hit) else stack.shape[1]
            texts.append(sync_card_masses(tok.decode(stack[b, :n].tolist())))
        return texts

    texts = []
    for s in range(0, len(rows), args.batch):
        chunk = rows[s:s + args.batch]
        texts.extend(run_batch(chunk))
        print(f"[steer] generated {min(s + args.batch, len(rows))}/{len(rows)}",
              flush=True)
    handle.remove()

    results = []
    for r, text in zip(rows, texts):
        parsed = parse_card(text)
        rec = {"point": list(r["point"]), "alpha": r["alpha"], "mode": r["mode"],
               "true_m0": r["point"][0], "true_lam": r["point"][1],
               "s_norm": float(np.linalg.norm(r["vec"])), **parsed,
               "n_chars": len(text)}
        results.append(rec)
        fn = os.path.join(OUT, f"card_{label_of(tuple(r['point']))}_"
                            f"a{r['alpha']:_g}_{r['mode']}.txt")
        with open(fn, "w") as f:
            f.write(text)
        print(f"[steer] {r['mode']:>7s} m{r['point'][0]:<3d} a={r['alpha']:>5g} "
              f"-> m0={parsed['m0']} lam={parsed['lam']}", flush=True)

    save_json({"results": results, "config": vars(args), "layer": layer},
              os.path.join(OUT, "summary.json"))
    plot_results(results, args)
    print(f"[steer] written {OUT}/summary.json", flush=True)


def plot_results(results, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phys = [r for r in results if r["mode"] == "physics" and r["m0"] is not None]
    if not phys:
        print("[steer] nothing parsable, skipping plot", flush=True)
        return
    masses = sorted({r["true_m0"] for r in phys})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap = plt.get_cmap("viridis")
    for m in masses:
        rs = sorted([r for r in phys if r["true_m0"] == m], key=lambda r: r["alpha"])
        x = [r["alpha"] for r in rs]
        y = [float(r["m0"]) for r in rs]
        col = cmap((m - 125) / 425)
        held = (m, 1) in [tuple(v) for v in HELDOUT]
        axes[0].plot(x, y, "o-", color=col, label=f"m={m}" + (" *" if held else ""))
        axes[0].axhline(m, color=col, ls=":", lw=0.8)
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("generated 25:m0")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("steering dose-response (held-out masses starred)")
    for r in results:
        if r["mode"] != "physics" or r["m0"] is None:
            continue
        col = cmap((r["true_m0"] - 125) / 425)
        axes[1].scatter(r["true_m0"], float(r["m0"]), color=col, s=14)
    for a in (2.0, 2.5):
        rs = [r for r in phys if r["alpha"] == a and r["m0"] is not None]
        if len(rs) >= 3:
            from scipy.stats import spearmanr

            sp = spearmanr([r["true_m0"] for r in rs], [float(r["m0"]) for r in rs]).statistic
            axes[1].scatter([r["true_m0"] for r in rs], [float(r["m0"]) for r in rs],
                            s=90, facecolors="none", edgecolors="k",
                            label=f"a={a:g} spearman {sp:.2f}")
    axes[1].plot([100, 575], [100, 575], "k--", lw=0.8)
    axes[1].set_xlabel("true m_Phi")
    axes[1].set_ylabel("generated 25:m0")
    axes[1].legend(fontsize=8)
    axes[1].set_title("generated vs true mass")
    lams = [r for r in phys if r["true_lam"] != 1 and r["lam"] is not None]
    for r in lams:
        col = "C2" if int(r["lam"]) == r["true_lam"] else "C3"
        axes[2].scatter(r["true_lam"], int(r["lam"]), color=col, s=14)
    axes[2].plot([0.5, 4.5], [0.5, 4.5], "k--", lw=0.8)
    axes[2].set_xlabel("true Lambda")
    axes[2].set_ylabel("generated HiddenValley:Lambda")
    axes[2].set_title(f"Lambda steering ({len(lams)} parsable rows, "
                     "green=correct)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "steer_response.png"), dpi=150)
    print(f"[steer] plot {OUT}/steer_response.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["fit", "probe", "steer", "all"])
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--llm-npz", default=LLM_NPZ)
    ap.add_argument("--variant", default="hv_only")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--layer", default="auto")
    ap.add_argument("--fit-set", default="all", choices=["all", "shells", "lam1"])
    ap.add_argument("--exclude-mass", type=int, default=0,
                    help="drop every point of this mass from all span-map fits")
    ap.add_argument("--prompt-mode", default="oneshot", choices=["oneshot", "instruction"])
    ap.add_argument("--steer-prompt", default="all",
                    choices=["all", "gen", "numspan"],
                    help="inject at all positions, generated tokens only, or "
                         "surgically at numeric spans (per-span vectors)")
    ap.add_argument("--alphas", default=DEFAULT_ALPHAS)
    ap.add_argument("--ctrl-alpha", type=float, default=2.5)
    ap.add_argument("--vec-source", default="fit", choices=["fit", "oracle"],
                    help="steer with the linear-map prediction (fit) or the true "
                         "card-representation difference (oracle)")
    ap.add_argument("--steer-set", default="lam1", choices=["lam1", "all"])
    ap.add_argument("--only", default="",
                    help="comma-separated masses to restrict steering to")
    ap.add_argument("--m0-map", default="mean", choices=["mean", "pos"],
                    help="m0/Lambda injection map: mean over digit span, or "
                         "per-digit-position maps")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.stage in ("fit", "all"):
        stage_fit(args)
    if args.stage in ("probe", "all"):
        stage_probe(args)
    if args.stage in ("steer", "all"):
        stage_steer(args)


if __name__ == "__main__":
    main()
