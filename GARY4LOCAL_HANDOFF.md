# Porting the SA3 backend to gary4local (x86 / py311)

Audience: the agent standing up the Stable Audio 3 backend inside gary4local.
This repo (`api.py` + `SA3_API_HANDOFF.md`) is the reference implementation —
gary4local will write its own `api.py`, so this note is just the stuff that
**isn't obvious from the code**: the dependency reality, the weights/gating
story, and what we added on top of stock SA3.

The engine is the `stable_audio_3` package from `~/stable-audio-3` (public-release
`main`). `api.py` imports it and serves async `/generate`, `/generate/loop`,
`/transform`, `/continue` (request/response contract in `SA3_API_HANDOFF.md`).

## Dependencies — x86/py311 is the easy path

The engine's `pyproject.toml` declares everything; on x86 it's painless:

- `requires-python >=3.10` → **py311 is fine**.
- It pins `torch==2.7.1` / `torchaudio==2.7.1` from a **cu126 wheel index gated on
  `platform_machine == 'x86_64'`**, so `uv sync` (or pip) pulls **prebuilt wheels**
  — no from-source build. (The Spark built torch cu128 + flash-attn from source
  *only* because it's ARM/Blackwell. Ignore all of that on x86.)
- `transformers>=5.8.0` (the T5Gemma text encoder) — a recent major; keep SA3 in
  its **own env**. You already isolate torch 2.7.1 + flash-attn per-env, so it slots in.
- `flash-attn` is **not** a hard dependency (optional perf) — use your existing
  x86 wheel if you want it; the engine runs without it.
- `pytorch_lightning` is the `lora` **training** extra — **not needed to serve**.
- `api.py` adds `flask` + `requests` (not in pyproject).
- Make `stable_audio_3` importable the normal way: `uv sync` / `pip install -e
  ~/stable-audio-3` in the checkout. (The Spark *vendors* the package via
  `sync_engine.sh` + `PYTHONPATH` to bake a self-contained image — a fresh machine
  pulling `main` doesn't need that.)

## Weights — canonical, NO shim

- Leave `SA3_USE_CACHED_PRERELEASE` **off** (default `0`). That shim exists *only*
  because the Spark's HF cache holds pre-release `…-ARC.*` files and the gated repo
  403s there — neither is true on a fresh machine.
- `medium` pulls the canonical files: `stabilityai/stable-audio-3-medium` →
  `model_config.json` + `model.safetensors` (~8.6 GB), plus the
  `google/t5gemma-b-b-ul2` encoder.
- **Both repos are gated.** One-time: accept the license on each model page, then
  `huggingface-cli login` (or set `HF_TOKEN`); the first run downloads + caches.
  (Our Spark 403 was an account that had lost gate access — make sure gary4local's
  HF account is on the access list for both.)
- Keep HF **online** for that first download. Only consider `HF_HUB_OFFLINE=1`
  later if you want to pin offline — the Spark sets it to dodge a t5gemma
  chat-template 404 at load, which only matters once everything's cached.
- Weights should match the Spark experience (`medium` = the ARC-distilled
  checkpoint promoted to canonical).

## What we added on top of stock SA3 (replicate these)

- **`latent_prefix` continuation mode** (our feature). Engine support = the
  `sample_flow_pingpong` patch in `stable_audio_3/inference/sampling.py` (already on
  `main`). Drive it via `model.generate(..., fixed_prefix_data=<encoded source
  latents>, fixed_prefix_mask=<latent mask, 1 = prefix tokens>)` with
  `sampler_type="pingpong"` (the only sampler that imposes the prefix). Orchestration
  reference: this repo's `/continue` worker in `api.py`, plus the standalone
  `~/stable-audio-3/test_latent_prefix.py`. Client contract: `SA3_API_HANDOFF.md`.
- **Serving loudness chain** (in `api.py`'s worker): adaptive latent rescale →
  tanh soft-knee true-peak limiter, plus the `/continue` `regen_past` tail. The
  blessed LoRA runs hot and clips long gens without it. Production-tuned env values
  are in the Spark compose `sa3` service — copy them as defaults:
  `SA3_CONTINUE_TAIL_MODE=regen_past`, `SA3_CONTINUE_TAIL_PAD=6`,
  `SA3_LATENT_RESCALE=1.0`, `SA3_LATENT_ADAPT_MIN=0.9`,
  `SA3_LIMITER_CEILING_DB=-0.3`, `SA3_LIMITER_KNEE=0.8`, `SA3_PEAK_NORMALIZE_DB=2.0`.
- **Pre-encode loudness matching** (`~/stable-audio-3/scripts/pre_encode_dataset.py
  --per_track_target_latent_rms`) — training-side; only relevant if gary4local
  retrains LoRAs, not for serving.

## LoRAs / prompts

`loras/` (`kev.ckpt`, `koan.ckpt`) + `prompts/` (`kev.json`, `koan.json`,
`defaults.json`) live in this repo — copy them over. The new engine loads `.ckpt`
LoRAs fine (just `torch.load` + `state_dict`).

## Spark → gary4local deltas

| | Spark (cloud) | gary4local |
|---|---|---|
| arch / python | ARM GB10 / 3.10 | x86_64 / 3.11 |
| torch | cu128, from source | cu126 prebuilt (`uv sync`) |
| flash-attn | from source (base image) | optional wheel |
| engine | vendored (`sync_engine.sh`) | `pip install -e ~/stable-audio-3` |
| weights | pre-release ARC + shim ON | canonical gated + shim OFF |
| HF | offline (cache primed) | online for first download |
| packaging | Docker image | local py311 process |

## Pointers

- Request/response contract (incl. `latent_prefix`): `SA3_API_HANDOFF.md`.
- Engine usage if importing directly instead of via `api.py`:
  `~/stable-audio-3/docs/workflows/inference.md`.
