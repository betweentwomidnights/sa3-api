#!/usr/bin/env python3
"""Record a stable_audio_3 generation with LoRA automation lanes, stage by stage, in the layout of
sa3.cpp's tools/dump_remote_parity.py, so sa3.cpp's tools/compare_remote_parity.py compares it
against `SA3_PARITY_DUMP=... sa3-generate --lane ...` (and --feed / SA3_PARITY_FEED teacher-forces).

The request is the HTTP contract, as JSON:

  {"prompt": "...", "duration": 10, "seed": 42, "steps": 8, "cfg_scale": 1.0,
   "loras": [{"name": "kev", "strength": 1.0}],
   "lora_lanes": [{"loras": [{"name": "keygen"}], "envelope": [[0,0],[10,1]], "prompt": "..."}],
   "lane_isolation": 1.0, "lane_isolation_until": 0.0, "lane_average_until": 0.9}

Run inside the sa3-api image (see LORA_LANES.md):
  python tools/dump_lane_parity.py --request req.json --out parity/ref
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def write_float_wav(path, x, sr):
    """[ch, n] float32 -> IEEE-float WAV (the image has no soundfile)."""
    import struct
    data = np.ascontiguousarray(x.T.astype("<f4")).tobytes()
    ch = x.shape[0]
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 3, ch, sr, sr * ch * 4, ch * 4, 32) + b"data" + struct.pack("<I", len(data))
    Path(path).write_bytes(hdr + data)


def install_prerelease_shim():
    # same remap as api.py (SA3_USE_CACHED_PRERELEASE=1)
    import stable_audio_3.models.conditioners as _cond
    import stable_audio_3.model_configs as _mc
    from huggingface_hub import hf_hub_download
    C = _cond.T5GemmaConditioner
    if "stabilityai/t5gemma-b-b-ul2" not in C.T5GEMMA_MODELS:
        C.T5GEMMA_MODELS = list(C.T5GEMMA_MODELS) + ["stabilityai/t5gemma-b-b-ul2"]
        C.T5GEMMA_MODEL_DIMS["stabilityai/t5gemma-b-b-ul2"] = C.T5GEMMA_MODEL_DIMS["google/t5gemma-b-b-ul2"]
    rev = os.environ.get("SA3_MEDIUM_REVISION", "95254c200f566987e3ad73492994fa7869d67f51")

    class Pinned:
        def resolve(self):
            return (hf_hub_download("stabilityai/stable-audio-3-medium", "stable-audio-3-medium-ARC.json", revision=rev),
                    hf_hub_download("stabilityai/stable-audio-3-medium", "stable-audio-3-medium-ARC.safetensors", revision=rev))
    _mc.all_models["medium"] = Pinned()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, help="JSON file in the /generate lane contract")
    ap.add_argument("--lora-dir", default="/app/loras")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--no-padding-mask", action="store_true",
                    help="diagnostic: withhold the attention padding mask (sa3.cpp has none)")
    ap.add_argument("--no-lanes", action="store_true", help="ignore lane fields (baseline run)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    if os.environ.get("SA3_USE_CACHED_PRERELEASE", "0") == "1":
        install_prerelease_shim()
    import stable_audio_3.inference.sampling as sampling
    from stable_audio_3 import StableAudioModel
    import lora_lanes

    req = json.loads(Path(args.request).read_text())
    prompt, duration = req.get("prompt", ""), float(req.get("duration", 10))
    steps, cfg, seed = int(req.get("steps", 8)), float(req.get("cfg_scale", 1.0)), int(req.get("seed", 42))

    files = sorted(f for f in os.listdir(args.lora_dir) if f.endswith((".ckpt", ".safetensors")))
    names = [os.path.splitext(f)[0] for f in files]

    def resolve_index(n):
        if n not in names:
            raise ValueError(f"unknown LoRA '{n}' (have {names})")
        return names.index(n)

    pipe = StableAudioModel.from_pretrained("medium", model_half=not args.fp32)
    pipe.load_lora([os.path.join(args.lora_dir, f) for f in files])
    shared = [{"lora_index": resolve_index(e["name"]), "name": e["name"],
               "strength": float(e.get("strength", 1.0)), "interval": (0.0, 1.0), "layer_filter": ""}
              for e in req.get("loras", [])]
    plan = None if args.no_lanes else lora_lanes.parse_lane_plan(req, shared, resolve_index, prompt)
    for i in range(len(names)):
        pipe.set_lora_strength(0.0, lora_index=i)
    for c in shared:
        pipe.set_lora_strength(c["strength"], lora_index=c["lora_index"])
    print("plan:", json.dumps(plan.describe() if plan else None))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, t):
        np.save(out / name, t.detach().float().cpu().numpy())

    rec = {"steps": 0}
    real_pingpong, real_randn_like = sampling.sample_flow_pingpong, torch.randn_like

    def pingpong(model, x, sigmas, **kw):
        save("noise0.npy", x[0])
        save("sigmas.npy", sigmas)

        def wrapped(xi, t, **mkw):
            i = rec["steps"]
            if args.no_padding_mask:
                mkw = {k: v for k, v in mkw.items() if k != "padding_mask"}
            v = model(xi, t, **mkw)
            save(f"step{i}_x.npy", xi[0])
            save(f"step{i}_v.npy", v[0])
            if i == 0:
                for key, val in mkw.items():
                    if isinstance(val, torch.Tensor):
                        save(f"{key}.npy", val[0])
            rec["steps"] = i + 1
            return v

        def randn_like(t, *a, **k):
            r = real_randn_like(t, *a, **k)
            save(f"step{rec['steps'] - 1}_noise.npy", r[0])
            return r

        torch.randn_like = randn_like
        try:
            return real_pingpong(wrapped, x, sigmas, **kw)
        finally:
            torch.randn_like = real_randn_like

    sampling.sample_flow_pingpong = pingpong
    gen = dict(prompt=prompt, negative_prompt=req.get("negative_prompt"), duration=duration, steps=steps,
               cfg_scale=cfg, seed=seed, sampler_type="pingpong", return_latents=True)
    if plan is None:
        gen["lora_configs"] = [{"lora_index": i, "interval": (0.0, 1.0), "layer_filter": ""}
                               for i in range(len(names))]
        latents = pipe.generate(**gen)
    else:
        with lora_lanes.lanes_active(pipe, plan, prompt, duration, len(names)) as run:
            latents = pipe.generate(**gen)
            np.save(out / "lane_envelopes.npy", np.stack(run.eb))
    sampling.sample_flow_pingpong = real_pingpong
    save("latent.npy", latents[0])

    if not args.no_decode:
        pt = pipe.model.pretransform
        audio = pt.decode(latents.to(dtype=next(pt.parameters()).dtype)).float().clamp(-1, 1)
        audio = audio[:, :, : int(duration * pipe.model.sample_rate)]
        save("audio.npy", audio[0])
        write_float_wav(out / "torch.wav", audio[0].cpu().numpy(), int(pipe.model.sample_rate))

    props = torch.cuda.get_device_properties(0)
    (out / "manifest.json").write_text(json.dumps({
        "request": req, "half": not args.fp32, "lanes": plan.describe() if plan else None,
        "latent_shape": list(latents.shape[1:]), "cuda": props.name,
        "rng_cuda_blocks": props.multi_processor_count * (props.max_threads_per_multi_processor // 256),
    }, indent=2))
    print(f"recorded {rec['steps']} steps, latent {tuple(latents.shape)} -> {out}")


if __name__ == "__main__":
    sys.exit(main())
