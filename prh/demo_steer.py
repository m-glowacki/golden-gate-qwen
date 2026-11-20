"""Interactive demo: steering Qwen3-8B Pythia-card generation with LHC events.

Golden-Gate-style pipeline, in one file:

    detector events -> frozen SSL encoder -> z (16-d)
                   -> linear maps W (one per numeric card setting)
                   -> residual-stream injection at layer 9, only while the
                      model is writing each number
                   -> the generated card's parameters follow the events.

The prompt is the Pythia card of the 125 GeV reference point; without steering
(alpha = 0) Qwen greedily copies it.  Selecting a target mass embeds that
point's detector events, injects W (z_target - z_125) at the numeric spans, and
the written numbers move toward the target.

Usage
-----
    python -m prh.demo_steer                     # web interface on :7860
    python -m prh.demo_steer --mass 550 --alpha 2.5
    python -m prh.demo_steer --mass 550 --alpha 2.5 --mode shuffle

Requires the cached artefacts of ``prh.steer_qwen --stage fit``:
``z_detector_two125_550_lam1_ssl_s0.npz`` and ``qwen_digit_states.npz`` under
``fisher_geometry/runs/prh_platonic/results/steer_qwen``.
"""

import argparse
import html
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .cards import card_variants  # noqa: E402

RUN = "fisher_geometry/runs/prh_platonic/results/steer_qwen"
ENCODER = ("fisher_geometry/runs/prh_platonic/encoders/detector/"
           "detector_two125_550_lam1_ssl_s0")
MODEL_DIR = "/eos/home-i03/m/maglowac/models/Qwen3-8B-Base"
REF = (125, 1)
LAYER = 9
HELDOUT = [(300, 1), (400, 1), (550, 1)]  # excluded from all map fits
MAX_NEW = 420

SPAN_RX = [(re.compile(r"25:m0\s*=\s*(\d*)$"), "m0"),
           (re.compile(r"m_phi\s*=\s*(\d*)$"), "m0"),
           (re.compile(r"HiddenValley:Lambda\s*=\s*(\d*)$"), "lam"),
           (re.compile(r"25:mMin\s*=\s*[\d.]*$"), "mmin"),
           (re.compile(r"25:mMax\s*=\s*[\d.]*$"), "mmax")]
NUM_LINE = re.compile(r"^.*(25:m0|25:mMin|25:mMax|HiddenValley:Lambda|m_phi =).*?$")


def _fit(X, H):
    X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
    return np.linalg.lstsq(X1, H, rcond=None)[0]


class SteerDemo:

    def __init__(self, encoder=ENCODER, layer=LAYER, device="cuda"):
        import torch

        self.torch = torch
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.layer = layer
        zpath = f"{RUN}/z_{encoder.split('/')[-1]}.npz"
        d = np.load(zpath, allow_pickle=True)
        self.pts = [tuple(int(v) for v in r) for r in d["points"]]
        self.Z = d["z"].astype(np.float64)
        self.iref = self.pts.index(REF)
        held = [self.pts.index(p) for p in HELDOUT]
        tr = [i for i in range(len(self.pts)) if i not in held]
        g = np.load(f"{RUN}/qwen_digit_states.npz", allow_pickle=True)
        self.W = {k: _fit(self.Z[tr], g[k].astype(np.float64)[tr, layer, :])
                  for k in ("m0", "lam", "mmin", "mmax")}
        rng = np.random.RandomState(7)
        self.perm = {k: rng.permutation(self.W[k].shape[1]) for k in self.W}
        self.rng = rng
        print(f"[demo] latents: {len(self.pts)} points, layer {layer}", flush=True)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16).to(self.dev)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.eos = self.tok.eos_token_id
        prompt = card_variants(REF)["hv_only"] + "\n"
        self.prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        self.state = {"vec": None}
        self.model.model.layers[layer - 1].register_forward_hook(self._hook)
        self._baseline = None

    def _hook(self, module, inputs, output):
        v = self.state["vec"]
        if v is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        hs.add_(v.to(hs.dtype).view(1, 1, -1))
        return output

    def span_vecs(self, point, mode):
        dz = self.Z[self.pts.index(point)] - self.Z[self.iref]
        vecs = {}
        for k, W in self.W.items():
            if mode == "shuffle":
                vecs[k] = dz @ W[:, self.perm[k]][:-1]
            elif mode == "random":
                vecs[k] = self.rng.standard_normal(W.shape[1]) * np.linalg.norm(dz @ W[:-1])
            else:
                vecs[k] = dz @ W[:-1]
        return vecs

    def generate(self, vecs, alpha):
        torch = self.torch
        with torch.no_grad():
            B = 1
            cur = {k: torch.tensor(alpha * v, device=self.dev, dtype=torch.float32)
                   for k, v in vecs.items()}
            ids = torch.tensor([self.prompt_ids], device=self.dev).expand(B, -1)
            out = self.model(ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
            rows, done, text = [nxt], nxt == self.eos, ""
            for _ in range(MAX_NEW - 1):
                nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
                text += self.tok.decode([int(nxt)])
                inject = None
                for rx, key in SPAN_RX:
                    if rx.search(text):
                        inject = cur[key]
                        break
                self.state["vec"] = inject if inject is not None else None
                out = self.model(nxt.unsqueeze(1), past_key_values=past,
                                 use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1].argmax(-1)
                done = done | (nxt == self.eos)
                rows.append(torch.where(done, torch.full_like(nxt, self.eos), nxt))
                if bool(done.all()):
                    break
            self.state["vec"] = None
            stack = torch.stack(rows, 1)[0]
            hit = (stack == self.eos).nonzero()
            n = int(hit[0]) if len(hit) else len(stack)
            return self.tok.decode(stack[:n].tolist())

    def baseline(self):
        if self._baseline is None:
            zero = {k: np.zeros(self.W[k].shape[1]) for k in self.W}
            self._baseline = sync_card_masses(self.generate(zero, 0.0))
        return self._baseline

    def run(self, mass, alpha, mode="physics"):
        point = (int(mass), 1)
        if point not in self.pts:
            raise ValueError(f"no latent for {point}")
        text = sync_card_masses(self.generate(self.span_vecs(point, mode), float(alpha)))
        return self.baseline(), text


# ----------------------------------------------------------------------------------------
# presentation helpers
# ----------------------------------------------------------------------------------------

def parse_card(text):
    out = {}
    for k, rx in (("m0", r"25:m0\s*=\s*(\d+)"), ("mmin", r"25:mMin\s*=\s*([\d.]+)"),
                  ("mmax", r"25:mMax\s*=\s*([\d.]+)"),
                  ("lam", r"HiddenValley:Lambda\s*=\s*(\d+)")):
        m = re.search(rx, text)
        out[k] = m.group(1) if m else None
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


def first_card(text):
    """The demo continuation writes several cards; keep the first."""
    head = text.split("\n! -")[0]
    return head if head.strip() else text[:600]


def render_html(text, color="#e8b4b8"):
    out = []
    for line in first_card(text).splitlines():
        esc = html.escape(line)
        if NUM_LINE.match(line):
            out.append(f'<span style="background:{color}">{esc}</span>')
        else:
            out.append(esc)
    return "<br>".join(out)


# ----------------------------------------------------------------------------------------
# CLI mode
# ----------------------------------------------------------------------------------------

def cli(args):
    demo = SteerDemo()
    base, text = demo.run(args.mass, args.alpha, args.mode)
    for name, t in (("BASELINE (alpha=0)", base), (f"STEERED m={args.mass} "
                   f"alpha={args.alpha} ({args.mode})", text)):
        print(f"\n=== {name} ===")
        print(first_card(t))
    print("\nparsed:", json.dumps(parse_card(text)))


# ----------------------------------------------------------------------------------------
# web interface (stdlib only)
# ----------------------------------------------------------------------------------------

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Golden Gate Qwen - LHC event steering</title>
<style>
 body{background:#14161a;color:#d6d8dc;font-family:system-ui;margin:2rem}
 h1{font-size:1.35rem;margin-bottom:.2rem}
 p.sub{color:#8a8f98;margin-top:0}
 .ctl{display:flex;gap:1.4rem;align-items:end;flex-wrap:wrap;margin:1.2rem 0}
 .ctl label{display:block;color:#8a8f98;font-size:.8rem;margin-bottom:.3rem}
 select,input[type=range]{accent-color:#e05d5d}
 button{background:#e05d5d;color:#fff;border:0;border-radius:6px;
        padding:.55rem 1.3rem;font-size:.95rem;cursor:pointer}
 button:disabled{background:#6a4a4a}
 .cols{display:flex;gap:1rem;flex-wrap:wrap}
 .card{flex:1;min-width:320px}
 .card h3{font-size:.85rem;color:#8a8f98}
 pre{background:#1d2026;border:1px solid #2c3037;border-radius:8px;
     padding:.9rem;font-size:.78rem;line-height:1.5;overflow-x:auto;white-space:pre-wrap}
 .parsed{font-family:monospace;font-size:.85rem;margin-top:.6rem;color:#e8b4b8}
 #msg{color:#8a8f98;font-style:italic}
</style></head><body>
<h1>Golden Gate Qwen &mdash; steering LLM generation with LHC events</h1>
<p class="sub">The prompt is the 125&nbsp;GeV Pythia card. Without steering Qwen copies it.
Pick a target mass: its detector events are embedded, the difference
<em>W&nbsp;(z<sub>target</sub>&nbsp;&minus;&nbsp;z<sub>125</sub>)</em> is injected at the
residual stream while each number is written, and the card's parameters move.</p>
<div class="ctl">
 <div><label>target mass (GeV)</label>
   <select id="mass">%%MASSES%%</select></div>
 <div><label>steering dose &alpha;</label>
   <input type="range" id="alpha" min="0" max="3" step="0.25" value="2.5">
   <span id="aval">2.5</span></div>
 <div><label>injected vector</label>
   <select id="mode">
     <option value="physics">physics (events)</option>
     <option value="shuffle">control: shuffled W</option>
     <option value="random">control: random direction</option>
   </select></div>
 <div><button id="go" onclick="run()">Generate card</button></div>
</div>
<div id="msg">model loading / ready</div>
<div class="cols">
 <div class="card"><h3>baseline &alpha;=0 &mdash; copy of the 125 GeV card</h3>
   <pre id="base">%%BASE%%</pre></div>
 <div class="card"><h3>steered generation</h3>
   <pre id="steer">press Generate</pre>
   <div class="parsed" id="parsed"></div></div>
</div>
<script>
const masses = %%MASSLIST%%;
const aval = document.getElementById('aval');
document.getElementById('alpha').oninput = e => aval.textContent = e.target.value;
async function run(){
  const b=document.getElementById('go'); b.disabled=true;
  document.getElementById('msg').textContent='generating (greedy, ~15 s)...';
  const body={mass:+document.getElementById('mass').value,
              alpha:+document.getElementById('alpha').value,
              mode:document.getElementById('mode').value};
  const r=await fetch('/generate',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();
  document.getElementById('steer').innerHTML=j.steered;
  document.getElementById('parsed').textContent='parsed: '+JSON.stringify(j.parsed);
  document.getElementById('msg').textContent='done';
  b.disabled=false;
}
</script></body></html>"""


def serve(args):
    demo = SteerDemo()
    masses = sorted({p[0] for p in demo.pts if p[1] == 1 and p != REF})
    base_html = render_html(demo.baseline(), color="#3a3f47")
    page = (PAGE.replace("%%MASSES%%", "".join(
        f'<option value="{m}"{" selected" if m == 550 else ""}>{m}</option>'
        for m in masses))
        .replace("%%MASSLIST%%", json.dumps(masses))
        .replace("%%BASE%%", base_html))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode())

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            try:
                _, text = demo.run(req["mass"], req["alpha"], req.get("mode", "physics"))
                resp = {"steered": render_html(text),
                        "parsed": parse_card(text)}
            except Exception as e:  # noqa: BLE001
                resp = {"error": str(e)}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())

        def log_message(self, *a):
            pass

    lock = threading.Lock()
    run_orig = demo.run

    def run_locked(*a, **k):
        with lock:
            return run_orig(*a, **k)

    demo.run = run_locked
    print(f"[demo] serving on http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mass", type=int, default=550)
    ap.add_argument("--alpha", type=float, default=2.5)
    ap.add_argument("--mode", default="physics",
                    choices=["physics", "shuffle", "random"])
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    if args.serve:
        serve(args)
    else:
        cli(args)


if __name__ == "__main__":
    main()
