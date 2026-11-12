# LoRA automation lanes (remote SA3 = local sa3.cpp contract)

`lora_lanes.py` ports sa3.cpp's automation lanes to this PyTorch service, so gary4juce (and the
mobile apps) send **one request shape** to remote SA3 and to local sa3.cpp. The contract,
defaults and error messages follow `sa3-server` (sa3.cpp `docs/SERVER.md`, "LoRA automation
lanes").

## Request fields (`/generate`, `/generate/loop`, `/transform`, `/continue`)

```json
"loras":      [{"name": "kev", "strength": 0.5}],
"lora_lanes": [
  {"loras": [{"name": "keygen", "strength": 0.8}], "envelope": [[0, 0], [4, 0], [6, 1.25], [10, 1]],
   "prompt": "demoscene chiptune, fast arpeggios"},
  {"loras": [{"name": "ratatat"}], "envelope": [[0, 1], [10, 0.3]]}
],
"lane_isolation": 1.0, "lane_isolation_until": 0.0, "lane_average_until": 0.9
```

- `envelope`: `[seconds, amount]` breakpoints, linear between points, held flat past either end.
  `amount` multiplies the adapter's `strength`. Time 0 is the start of the canvas (for
  `/continue`, the start of the kept source). Limits: 8 lanes, 256 points, amount 0..100.
- `loras` (request level) are **shared**: they run in every pass and are the only adapters that
  shape the seconds_total conditioner token.
- Each lane is one extra DiT pass per step. Per latent frame the velocity is
  `v_base + Σ e_k(t)·(v_k − v_base)`, mixed separately for the conditioned and unconditioned
  branches before CFG.
- `lora_envelope` + `loras` with no `lora_lanes`: the one-lane shorthand (every request LoRA moves
  on that curve; nothing shared).
- `lane_average_until` / `lane_isolation` / `lane_isolation_until`: as in sa3.cpp. API defaults
  are 0 / 0 / 0.9 (off). gary4juce's local path (C ABI `request_init`) uses **0.9 / 1 / 0**, so the
  plugin should send those explicitly to get the same sound remotely.
- Omitting every lane field leaves the request on the old code path, byte for byte.
- `/health.capabilities.lora_lanes` advertises support. Completed polls carry `meta.lora_lanes`.

## Differences from sa3.cpp (rejected with HTTP 400, never silently different)

| case | why |
| --- | --- |
| `lora_envelope_mode: "layer"` | not ported (velocity mode only) |
| an adapter both shared and in a lane, or twice in one lane | torch parametrizations hold one strength per adapter, so it can't be stacked with itself |
| shared LoRA with a non-default `interval_*` / `layer_filter` while lanes are on | no sa3.cpp equivalent |

One subtlety: when a single pass holds more than one DoRA (shared + lane), Python applies them
in registry (file-name) order, while sa3.cpp applies shared then lane order. DoRA renormalizes,
so the order can matter slightly. The parity case below used a shared LoRA that sorts first, so
it didn't exercise this.

## Cost

Each lane adds one DiT pass per step (CFG doubles every pass). DoRA weights are rebuilt on every
forward (~2× the DiT forward per active adapter on the Spark), so each pass's effective weights
are cached for the whole generation, which is bit-identical. Budget: `SA3_LANE_WEIGHT_CACHE_GB`
(default 12, about 2.7 GB per adapted pass in fp16, so 4 passes; 0 disables). Passes past the budget are
recomputed every step.

## Parity with sa3.cpp (2026-11-07, DGX Spark, seed-matched Philox noise)

Recorded with `tools/dump_lane_parity.py` and compared with sa3.cpp's
`tools/compare_remote_parity.py`, teacher-forced (`--feed steps` + `SA3_PARITY_FEED`), so each
step's velocity is compared on identical input. The reference ran in fp32 without the padding
mask, which removes the known backend floor (TORCH_PARITY.md). With fp16 plus the padding mask,
even the base model differs by 6–16% per step.

| case | per-step v rel | final latent |
| --- | --- | --- |
| base model, no LoRA (the floor) | 0.34–2.9% | cos 0.999999, rel 0.14% |
| 2 lanes kev→keygen crossfade, cfg 1 | 0.29–1.7% | cos 0.999999, rel 0.12% |
| shared kev + lane prompt + averaging + isolation, cfg 1 | 0.37–4.2% | cos 0.999999, rel 0.15% |
| same, cfg 2 + negative prompt (with the sa3.cpp fix below) | 0.9–5.3% | cos 0.999994, rel 0.35% |

For scale, the lanes move the latent by 71% relative to the same seed without lanes.

**sa3.cpp bug found on the way:** with a negative prompt, stable_audio_3's CFG branch zeroes the
negative prompt's padding tokens (`torch.where(negative_cross_attn_mask, neg, 0)`), but sa3.cpp
kept the padding embedding there. That alone put CFG + negative prompt at 3.4% latent rel (up to
45% per step) with no lanes. The one-loop fix in `src/sa3_pipeline.h` (uncond_crossb) brings it to
0.24%.

### Re-running

```bash
# Python reference (inside the sa3-api image; registry LoRAs from /app/loras)
python tools/dump_lane_parity.py --request req.json --out ref --fp32 --no-padding-mask --no-decode
# sa3.cpp, same seed and the same lanes on the CLI, teacher-forced on the reference's states
python tools/compare_remote_parity.py --ref ref --feed feed --feed-only steps
SA3_PARITY_FEED=feed SA3_PARITY_DUMP=cpp sa3-generate --model medium --rng torch-cuda \
  --lane keygen --lane-envelope 0:0,10:1 ... --no-peak-normalize --no-limiter --out cpp/cpp.wav
python tools/compare_remote_parity.py --ref ref --cpp cpp
```
