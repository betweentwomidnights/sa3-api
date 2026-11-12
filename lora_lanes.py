"""LoRA automation lanes for the stable_audio_3 (PyTorch) backend.

A port of sa3.cpp's automation lanes (src/sa3_pipeline.h, docs/SERVER.md "LoRA automation
lanes") so that gary4juce's remote SA3 (this service) and local SA3 (sa3.cpp) accept the same
request and produce the same clip for the same seed.

Request contract (identical to sa3-server):

    "loras":      [{"name": "kev", "strength": 1.0}, ...]     # shared: in EVERY pass
    "lora_lanes": [{"loras": [{"name": "keygen", "strength": 1.0}],
                    "envelope": [[sec, amount], ...],
                    "prompt": "optional lane prompt"}, ...]   # <= 8 lanes, <= 256 points
    "lora_envelope": [[sec, amount], ...]   # one-lane shorthand: every request LoRA on one curve
    "lora_envelope_mode": "velocity"        # "layer" is sa3.cpp-only (rejected here)
    "lane_isolation": 0.0, "lane_isolation_until": 0.9, "lane_average_until": 0.0

Semantics, all mirrored from sa3.cpp:

  * Breakpoints are linear between points and held flat past either end, sampled at every
    latent frame's centre ((i + 0.5) * 4096 / 44100 s) in float32, divided by the lane's peak.
  * Pass 0 runs the shared adapters alone on the request prompt; pass k runs the shared adapters
    plus lane k's adapters at strength * peak, on the lane's prompt when it has its own. Per frame
    the velocity is v0 + sum_k e_k(t) * (v_k - v0). Under CFG this mix is done separately for the
    conditioned and the unconditioned branch (the unconditioned branch keeps the negative/null
    conditioning for every pass), and only then guided, exactly as sa3.cpp does.
  * Lanes whose curve is all zero, or with no adapters and no prompt of their own, are dropped.
  * The seconds_total conditioner token is adapted by the REQUEST-level loras only (at their
    strength); lane adapters stay on the DiT.
  * lane_average_until: while the step's sigma is above it, every lane plays at the mean of its
    curve over the canvas (isolation is paused for those steps).
  * lane_isolation: while sigma > lane_isolation_until (and not averaged), self-attention between
    frames i and j is biased by -lane_isolation * sum_k |e_k(i) - e_k(j)|; memory/prepend tokens
    are unbiased. Applied to every pass.

Python-side limitations (rejected with a clear error rather than silently diverging):
  * an adapter cannot be both shared and in a lane, or twice in one lane (torch parametrizations
    hold one strength per adapter, so it cannot be stacked with itself);
  * "layer" envelope mode is not implemented;
  * shared LoRAs must use the default interval (0..1) and no layer_filter while lanes are active.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

MAX_LANES = 8
MAX_ENVELOPE_POINTS = 256
MAX_LANE_AMOUNT = 100.0
LANE_FIELDS = ("lora_lanes", "lora_envelope", "lora_envelope_mode",
               "lane_isolation", "lane_isolation_until", "lane_average_until")


class LaneError(ValueError):
    """A request-level problem (HTTP 400)."""


@dataclass
class Lane:
    loras: list            # [(lora_index, strength)]
    envelope: list         # [(seconds, amount)]
    prompt: str = ""
    peak: float = 0.0
    names: list = field(default_factory=list)


@dataclass
class LanePlan:
    shared: list           # [(lora_index, strength)] — in every pass
    lanes: list            # [Lane]
    isolation: float = 0.0
    isolation_until: float = 0.9
    average_until: float = 0.0

    def describe(self):
        return {
            "lanes": [{"loras": [[n, s] for n, (_, s) in zip(ln.names, ln.loras)],
                       "points": len(ln.envelope), "peak": ln.peak,
                       "prompt": ln.prompt or None} for ln in self.lanes],
            "lane_isolation": self.isolation,
            "lane_isolation_until": self.isolation_until,
            "lane_average_until": self.average_until,
        }


# --------------------------------------------------------------------------- request parsing

def wants_lanes(data: dict) -> bool:
    return bool(data.get("lora_lanes")) or bool(data.get("lora_envelope"))


def _finite(v, what):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise LaneError(f"{what} must be a number")
    if not math.isfinite(f):
        raise LaneError(f"{what} must be finite")
    return f


def _points(raw, what):
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LaneError(f"{what}: expected [[seconds, amount], ...]")
    if len(raw) > MAX_ENVELOPE_POINTS:
        raise LaneError(f"{what}: at most {MAX_ENVELOPE_POINTS} envelope points")
    pts = []
    for bp in raw:
        if not isinstance(bp, (list, tuple)) or len(bp) != 2:
            raise LaneError(f"{what}: expected [[seconds, amount], ...]")
        t, a = _finite(bp[0], what), _finite(bp[1], what)
        if a < 0.0 or a > MAX_LANE_AMOUNT:
            raise LaneError(f"{what}: amounts must be finite and in [0, {MAX_LANE_AMOUNT:g}]")
        pts.append((t, a))
    return pts


def parse_lane_plan(data: dict, shared_loras: list, resolve_index, base_prompt: str) -> LanePlan | None:
    """Validate the lane fields of a request; None when it asks for no automation.

    shared_loras: resolve_loras() output for the request's "loras" ([{lora_index, name, strength,
    interval, layer_filter}]). resolve_index(name) -> registry index or raises ValueError.
    """
    if not wants_lanes(data):
        return None
    mode = data.get("lora_envelope_mode", "velocity")
    if mode not in ("velocity", "layer"):
        raise LaneError("lora envelope: mode must be 'velocity' or 'layer'")
    if mode == "layer":
        raise LaneError("lora envelope: 'layer' mode is not supported by this backend; use 'velocity'")
    iso = _finite(data.get("lane_isolation", 0.0), "lane_isolation")
    if iso < 0.0:
        raise LaneError("lane_isolation must be a finite number >= 0")
    iso_until = _finite(data.get("lane_isolation_until", 0.9), "lane_isolation_until")
    avg_until = _finite(data.get("lane_average_until", 0.0), "lane_average_until")

    for c in shared_loras:
        if tuple(c.get("interval", (0.0, 1.0))) != (0.0, 1.0) or c.get("layer_filter"):
            raise LaneError(f"LoRA '{c['name']}': interval/layer_filter are not supported with lanes")
    shared = [(c["lora_index"], float(c["strength"])) for c in shared_loras]
    shared_names = [c["name"] for c in shared_loras]

    env = _points(data.get("lora_envelope"), "lora envelope")
    raw_lanes = data.get("lora_lanes") or []
    if not isinstance(raw_lanes, list):
        raise LaneError("lora_lanes must be a list")
    if len(raw_lanes) > MAX_LANES:
        raise LaneError(f"lora lanes: at most {MAX_LANES} lanes")

    lanes = []
    for i, rl in enumerate(raw_lanes):
        if not isinstance(rl, dict):
            raise LaneError("lora_lanes: each lane must be an object")
        pts = _points(rl.get("envelope"), "lora lane")
        if not pts:
            raise LaneError("lora_lanes: each lane needs an envelope")
        lane_loras, names = [], []
        for e in rl.get("loras") or []:
            if not isinstance(e, dict) or not e.get("name"):
                raise LaneError("lora_lanes: each lora needs a name")
            name = str(e["name"])
            strength = _finite(e.get("strength", 1.0), "lora lane: adapter strengths")
            try:
                idx = resolve_index(name)
            except ValueError as exc:
                raise LaneError(str(exc))
            if idx in [j for j, _ in shared]:
                raise LaneError(f"lora lane {i}: '{name}' is also a shared LoRA; this backend "
                                "cannot stack an adapter with itself")
            if idx in [j for j, _ in lane_loras]:
                raise LaneError(f"lora lane {i}: '{name}' listed twice")
            lane_loras.append((idx, strength))
            names.append(name)
        prompt = rl.get("prompt") if isinstance(rl.get("prompt"), str) else ""
        lanes.append(Lane(lane_loras, pts, prompt, names=names))

    if not lanes and env and shared:
        # A lone lora_envelope is one lane carrying every DiT adapter, with nothing shared.
        lanes = [Lane(list(shared), env, "", names=list(shared_names))]
        shared = []

    for ln in lanes:
        ln.peak = max(0.0, max(a for _, a in ln.envelope))
    lanes = [ln for ln in lanes
             if ln.peak > 0.0 and not (not ln.loras and (not ln.prompt or ln.prompt == base_prompt))]
    if not lanes:
        return None    # nothing can differ from the base pass: an ordinary generation
    return LanePlan(shared, lanes, iso, iso_until, avg_until)


# --------------------------------------------------------------------------- envelope math

def sample_envelope(bps, peak, T, samples_per_frame=4096, sample_rate=44100.0):
    """sa3.cpp's sample_envelope, in float32 with its operation order."""
    pts = sorted(bps)
    f32 = np.float32
    spf, sr, pk = f32(samples_per_frame), f32(sample_rate), f32(peak)
    e = np.empty(T, dtype=np.float32)
    for i in range(T):
        t = (f32(i) + f32(0.5)) * spf / sr
        a = f32(pts[-1][1])
        if t <= f32(pts[0][0]):
            a = f32(pts[0][1])
        else:
            for k in range(1, len(pts)):
                if t <= f32(pts[k][0]):
                    t0, t1 = f32(pts[k - 1][0]), f32(pts[k][0])
                    u = (t - t0) / (t1 - t0) if t1 > t0 else f32(1.0)
                    a = f32(pts[k - 1][1]) + u * (f32(pts[k][1]) - f32(pts[k - 1][1]))
                    break
        e[i] = a / pk
    return e


# --------------------------------------------------------------------------- engine hooks

def _find_dit(pipe):
    from stable_audio_3.models.dit import DiffusionTransformer
    for m in pipe.model.model.modules():
        if isinstance(m, DiffusionTransformer):
            return m
    raise RuntimeError("lora lanes: no DiffusionTransformer in the model")


def _lora_params_by_index(module):
    from stable_audio_3.models.lora.model import LoRAParametrization
    by = {}
    for m in module.modules():
        if isinstance(m, LoRAParametrization):
            by.setdefault(m.lora_index, []).append(m)
    return by


def _check_functional(params_by_index, indices):
    for idx in indices:
        for p in params_by_index.get(idx, []):
            if p.adapter_type not in ("lora", "dora-rows", "dora"):
                raise LaneError("lora lanes: every lane adapter must be a 'lora' or 'dora-rows' "
                                f"adapter (index {idx} is '{p.adapter_type}')")


class _Runner:
    """Per-generation state: owns the hooks installed on the DiT."""

    def __init__(self, pipe, plan: LanePlan, lane_cross, n_registry: int):
        self.pipe, self.plan, self.lane_cross = pipe, plan, lane_cross
        self.dit = _find_dit(pipe)
        self.by_index = _lora_params_by_index(self.dit)
        self.registry = range(n_registry)
        self.spf = int(getattr(pipe.model.pretransform, "downsampling_ratio", 4096))
        self.sr = float(pipe.model.sample_rate)
        self._env_T = None
        self.batch = None
        self.sigma = None
        self.iso_bias = None      # [T, T] float32 when isolation is armed for this step
        self.iso_active = False
        self.passes = [list(plan.shared)]
        for ln in plan.lanes:
            self.passes.append(list(plan.shared) + [(i, s * ln.peak) for i, s in ln.loras])
        _check_functional(self.by_index, {i for p in self.passes for i, _ in p})
        import os
        self.cache_budget = float(os.environ.get("SA3_LANE_WEIGHT_CACHE_GB", "12")) * 2**30
        self.cache_on = self.cache_budget > 0
        self.wcache, self.uncached, self.cache_bytes, self._filling = {}, set(), 0, False
        # storage of the unadapted weights: a pass whose adapters are all at strength 0 returns
        # these tensors themselves, which costs no memory
        self.base_ptrs = {p.data_ptr() for p in self.dit.parameters()}

    # -- envelopes, built once per canvas length
    def envelopes(self, T):
        if self._env_T != T:
            self.eb = [sample_envelope(ln.envelope, ln.peak, T, self.spf, self.sr)
                       for ln in self.plan.lanes]
            self.eb_avg = [np.full(T, np.float32(e.astype(np.float64).mean()), dtype=np.float32)
                           for e in self.eb]
            if self.plan.isolation > 0.0:
                d = np.zeros((T, T), dtype=np.float32)
                for e in self.eb:
                    d += np.abs(e[:, None] - e[None, :])
                self._iso_full = torch.from_numpy(np.float32(-self.plan.isolation) * d)
            self._env_T = T
        return self.eb, self.eb_avg

    # -- per-pass effective-weight cache. A parametrized (DoRA) weight is rebuilt on every forward,
    # ~2x the cost of the DiT forward itself per active adapter, yet a pass's weights never change
    # across steps. torch.nn.utils.parametrize keeps a module-global cache dict while caching is
    # enabled; giving each pass its own dict means torch fills it on the pass's first step and
    # reuses the very same tensors afterwards (bit-identical). Budget-capped: a pass that would push
    # the total past it is recomputed every step instead.
    def _use_weight_cache(self, k):
        if not self.cache_on:
            return
        import torch.nn.utils.parametrize as P
        P._cache = self.wcache[k] if k in self.wcache else {}
        self._filling = k not in self.wcache and k not in self.uncached

    def _account_weight_cache(self, k):
        if not self.cache_on or not self._filling:
            return
        import torch.nn.utils.parametrize as P
        new = sum(t.numel() * t.element_size() for t in P._cache.values()
                  if t is not None and t.data_ptr() not in self.base_ptrs)
        if self.cache_bytes + new <= self.cache_budget:
            self.wcache[k] = P._cache
            self.cache_bytes += new
        else:
            self.uncached.add(k)
        P._cache = {}

    def set_strengths(self, pairs):
        want = dict(pairs)
        for idx in self.registry:
            s = float(want.get(idx, 0.0))
            for p in self.by_index.get(idx, []):
                p.lora_strength.fill_(s)

    # -- DiffusionTransformer.forward wrapper: records batch size and this step's sigma
    def forward(self, real):
        def fwd(x, t, *a, **kw):
            self.batch = x.shape[0]
            self.sigma = float(t.float().reshape(-1)[0])
            kw.pop("lora_configs", None)
            return real(x, t, *a, **kw)
        return fwd

    # -- DiffusionTransformer._forward wrapper: one call per pass, then the per-frame mix
    def _forward(self, real):
        def fwd(x, t, *a, cross_attn_cond=None, **kw):
            plan, T = self.plan, x.shape[-1]
            eb, eb_avg = self.envelopes(T)
            averaged = plan.average_until > 0.0 and self.sigma > plan.average_until
            curves = eb_avg if averaged else eb
            self.iso_active = (plan.isolation > 0.0 and not averaged
                               and self.sigma > plan.isolation_until)
            self._mask_cache = {}
            if self.iso_active:
                self.iso_bias = self._iso_full.to(x.device)
            n_cond = self.batch if self.batch is not None else x.shape[0]
            try:
                outs = []
                for k, pairs in enumerate(self.passes):
                    self.set_strengths(pairs)
                    self._use_weight_cache(k)
                    cross = cross_attn_cond
                    if k > 0 and cross is not None:
                        lc = self.lane_cross.get(plan.lanes[k - 1].prompt)
                        if lc is not None:
                            cross = cross.clone()
                            cross[:n_cond] = lc.to(cross.dtype).expand(n_cond, -1, -1)
                    outs.append(real(x, t, *a, cross_attn_cond=cross, **kw))
                    self._account_weight_cache(k)
            finally:
                self.iso_active = False
                self.iso_bias = None
                self._mask_cache = {}
            v0 = outs[0].float()
            acc = v0.clone()
            for vk, e in zip(outs[1:], curves):
                ek = torch.from_numpy(e).to(acc.device)
                acc += (vk.float() - v0) * ek
            return acc.to(outs[0].dtype)
        return fwd

    # -- self-attention: SDPA with the isolation bias (+ the key padding mask) when armed
    def apply_attn(self, attn, real):
        def fn(q, k, v, causal=None, flex_attention_block_mask=None, flex_attention_score_mod=None,
               flash_attn_sliding_window=None, padding_mask=None, varlen_metadata=None):
            if not self.iso_active:
                return real(q, k, v, causal=causal, flex_attention_block_mask=flex_attention_block_mask,
                            flex_attention_score_mod=flex_attention_score_mod,
                            flash_attn_sliding_window=flash_attn_sliding_window,
                            padding_mask=padding_mask, varlen_metadata=varlen_metadata)
            if flash_attn_sliding_window is not None or flex_attention_block_mask is not None:
                raise RuntimeError("lane isolation: sliding-window/block-mask attention is not supported")
            if attn.num_heads != attn.kv_heads:
                r = attn.num_heads // attn.kv_heads
                k, v = k.repeat_interleave(r, dim=1), v.repeat_interleave(r, dim=1)
            # Built once per step: within one _forward call every pass and layer sees the same
            # padding mask (the cache is cleared at the start of each step).
            S, T = q.shape[2], self.iso_bias.shape[0]
            key = (S, q.dtype, q.shape[0], padding_mask is not None)
            m = self._mask_cache.get(key)
            if m is None:
                P = S - T                               # memory + prepend tokens: unbiased
                m = torch.zeros((S, S), dtype=torch.float32, device=q.device)
                m[P:, P:] = self.iso_bias
                m = m.to(q.dtype).view(1, 1, S, S)
                if padding_mask is not None:
                    m = m.expand(q.shape[0], 1, S, S).masked_fill(
                        ~padding_mask.bool()[:, None, None, :], float("-inf"))
                self._mask_cache[key] = m
            return F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=False)
        return fn


def encode_lane_prompts(pipe, plan: LanePlan, base_prompt: str, seconds_total: float):
    """cross_attn_cond for each distinct lane prompt, with the strengths currently set (= the
    request loras, which is what adapts the seconds token in sa3.cpp)."""
    out = {}
    model = pipe.model
    device = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype
    for ln in plan.lanes:
        p = ln.prompt
        if not p or p == base_prompt or p in out:
            continue
        ct = model.conditioner([{"prompt": p, "seconds_total": seconds_total}], device)
        parts = []
        for key in model.cross_attn_cond_ids:
            c, _ = ct[key]
            parts.append(c.unsqueeze(1) if c.dim() == 2 else c)
        out[p] = torch.cat(parts, dim=1).to(dtype)
    return out


@contextmanager
def lanes_active(pipe, plan: LanePlan, base_prompt: str, seconds_total: float, n_registry: int):
    """Install the lane hooks for one generate() call.

    The caller must already have set every registry LoRA to its REQUEST strength (shared loras at
    their strength, everything else 0) and must call generate() with lora_configs=None, so the
    conditioner (seconds token) sees exactly the request loras, as in sa3.cpp."""
    lane_cross = encode_lane_prompts(pipe, plan, base_prompt, seconds_total)
    run = _Runner(pipe, plan, lane_cross, n_registry)
    dit = run.dit
    saved = {"forward": dit.__dict__.get("forward"), "_forward": dit.__dict__.get("_forward")}
    dit.forward = run.forward(dit.forward)
    dit._forward = run._forward(dit._forward)
    attns = []
    if plan.isolation > 0.0:
        for layer in dit.transformer.layers:
            a = getattr(layer, "self_attn", None)
            if a is not None:
                attns.append((a, a.__dict__.get("apply_attn")))
                a.apply_attn = run.apply_attn(a, a.apply_attn)
    import torch.nn.utils.parametrize as P
    prev_cache_enabled, prev_cache = P._cache_enabled, P._cache
    if run.cache_on:
        P._cache_enabled += 1
    try:
        yield run
    finally:
        if run.cache_on:
            P._cache_enabled, P._cache = prev_cache_enabled, prev_cache
        if run.wcache:
            print(f"lora lanes: cached {len(run.wcache)} pass weight set(s), "
                  f"{run.cache_bytes / 2**30:.2f} GB; recomputed {len(run.uncached)}")
        run.wcache.clear()
        for name, prev in saved.items():
            if prev is None:
                dit.__dict__.pop(name, None)
            else:
                setattr(dit, name, prev)
        for a, prev in attns:
            if prev is None:
                a.__dict__.pop("apply_attn", None)
            else:
                a.apply_attn = prev
