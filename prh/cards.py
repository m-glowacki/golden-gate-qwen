"""Pythia card loading and controlled variants for theory representations."""

from __future__ import annotations

import os
from pathlib import Path

from .splits import ALL_POINTS

PROC_ROOT = "/eos/home-i03/m/maglowac/info_geometries/samples/collide_simulation/processes"


def point_to_dir(point):
    m, l = int(point[0]), int(point[1])
    return f"SUEPlike_HV_mPhi{m}_mX2_Lam{l}"


def card_path(point, root=PROC_ROOT):
    d = Path(root) / point_to_dir(point)
    cand = d / f"{point_to_dir(point)}_pythia_card.dat"
    if cand.exists():
        return cand
    hits = sorted(d.glob("*pythia_card*.dat"))
    if not hits:
        raise FileNotFoundError(f"no pythia card for {point} under {d}")
    return hits[0]


def read_card(point, root=PROC_ROOT):
    return card_path(point, root).read_text()


HV_PREFIXES = ("HiddenValley:", "4900", "25:", "HiggsSM:", "PhaseSpace:")


def is_hv_relevant(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("!"):
        return False
    key = s.split("=", 1)[0].strip().split()[0]
    return key.startswith(HV_PREFIXES)


def hv_only_card(text: str) -> str:
    kept = []
    header = []
    in_header = True
    for line in text.splitlines():
        s = line.strip()
        if in_header and s.startswith("!"):
            header.append(line)
            continue
        in_header = False
        if is_hv_relevant(line):
            kept.append(line.strip())
    out = ["! HV-only settings extracted from Pythia card"]
    out.extend([h for h in header[:4]])
    out.extend(kept)
    return "\n".join(out) + "\n"


def normalize_serialization(text: str) -> str:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("!"):
            out.append("!" + " " + s.lstrip("!").strip())
            continue
        if "=" in s:
            left, right = s.split("=", 1)
            left = " ".join(left.split())
            right = " ".join(right.split())
            out.append(f"{left} = {right}")
        else:
            out.append(" ".join(s.split()))
    return "\n".join(out) + "\n"


def reorder_settings(text: str) -> str:
    lines = text.splitlines()
    header, settings, seen = [], [], False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if not seen and s.startswith("!"):
            header.append(s)
            continue
        seen = True
        if s.startswith("!"):
            continue
        settings.append(s)
    settings = sorted(set(settings))
    return "\n".join(header + settings) + "\n"


def param_summary(point):
    m, l = int(point[0]), int(point[1])
    return (
        f"Hidden Valley Higgs-portal SUEP-like model. "
        f"Mediator mass m_Phi = {m} GeV. Confinement scale Lambda = {l} GeV. "
        f"Dark quark mass m_X = 2 GeV. nFlav = 2, Ngauge = 3."
    )


def card_variants(point, root=PROC_ROOT):
    full = read_card(point, root)
    hv = hv_only_card(full)
    return {
        "full": full,
        "hv_only": hv,
        "full_normalized": normalize_serialization(full),
        "full_reordered": reorder_settings(full),
        "hv_normalized": normalize_serialization(hv),
        "param_summary": param_summary(point),
    }


def all_card_variants(points=None, root=PROC_ROOT):
    pts = list(points) if points is not None else list(ALL_POINTS)
    return {tuple(map(int, p)): card_variants(tuple(map(int, p)), root) for p in pts}
