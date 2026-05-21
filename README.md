# sa3 — Stable Audio 3 API

VST-facing async API serving the `medium` (ARC-distilled) model + a default
LoRA to the gary4local VST, from the `gary-backend-spark` compose network.
Companion to `~/stable-audio-api`; mirrors the foundation-1 async/polling
contract the JUCE frontend already speaks.

## Architecture

- **Image**: `FROM sa3:spark` (deps-only base — torch cu128, flash-attn,
  transformers; no source baked) + `flask`/`requests` + **vendored**
  current-`main` `stable_audio_3/` source. Not `sa3:spark-lightning` —
  Lightning is training-only; `.ckpt` LoRA load is just `torch.load`.
- **GPU serialization**: dedicated `sa3` lane (concurrency=1) in
  `gary-backend-spark/gpu-queue-service`, a mirror of the foundation lane.
- **No auth**. Single-user beta; the backend queue is the gate.

## v1 endpoints

| Method | Path                   | Purpose                                  |
|--------|------------------------|------------------------------------------|
| POST   | `/generate`            | text→audio, async → `session_id`         |
| POST   | `/generate/loop`       | bar-aligned loop (BPM from prompt), async |
| POST   | `/transform`           | init_audio style transfer, output = input length, async |
| POST   | `/continue`            | inpaint continuation of DAW source, async |
| GET    | `/prompts`             | dice-button pool (live-editable, `?lora=<name>`) |
| GET    | `/poll_status/<id>`    | progress + base64 WAV (JUCE contract)    |
| GET    | `/health`, `/ready`    | model warm state                         |

### `/prompts` (dice button)

`GET /prompts[?lora=<name>]` → `{success, lora, available_loras,
prompts:{version, dice:{generic,instrumental,drums,...}, source}}` (the
`~/stable-audio-api` JUCE contract). Pools live in `~/sa3/prompts/` —
**bind-mounted, read fresh every request**: edit `defaults.json` (generic, no
LoRA) or `<lora>.json` on the host and the next dice roll reflects it, no
rebuild/restart, no plugin reship. `?lora=` is repeatable / comma-separated
(`?lora=a,b` or `?lora=a&lora=b`) — when the UI has >1 LoRA slider up, each
bucket those LoRAs define becomes the **deduped union** across all of them
(one roll can land in either LoRA's training distribution), replacing the
generic default for that bucket; other buckets stay generic. Response:
`{success, loras, missing_loras, available_loras, prompts:{version,dice,source}}`.

Never put bpm/key in dice prompts — the VST appends them. Seed a new LoRA's
pool from its SA3 training captions:

```bash
./build_lora_prompts.py --name <lora> --captions-dir <dir of clip .txt>
# strips ", <bpm> bpm, <key>", dedups genres; won't clobber a curated file (--force)
```

### `/continue` body

Same fields as `/generate` **except no `duration`**, plus:

| field                  | default   | notes |
|------------------------|-----------|-------|
| `audio_data`           | — (req)   | base64 WAV — the source to continue |
| `continuation_seconds` | `8.0`     | new audio generated after the source |
| `continuation_mode`    | `inpaint` | `latent_prefix` reserved (not yet implemented → 400) |

Keeps `[0, source]`, regenerates `[source, source+continuation]` toward the
prompt ("fill forward"). Output length = source + continuation, sample-exact.
The kept region is re-encoded through the autoencoder — musically continuous,
not a bit-identical sample-join. Saved as `sa3cont_<sid>_<seed>.wav`;
`poll_status.meta.continue` carries the mask/duration breakdown.

### `/transform` body

Same fields as `/generate` **except no `duration`** (it's the input's length), plus:

| field        | default | notes |
|--------------|---------|-------|
| `audio_data` | — (req) | base64-encoded WAV recorded in the DAW |
| `strength`   | `0.9`   | `init_noise_level`, 0.01–1.0; 0.01 ≈ preserve input, 1.0 ≈ full transform |

Output is the **exact same length (and channel count)** as the input —
sample-exact round-trip so it lines up in the DAW. Saved as
`sa3xform_<sid>_<seed>.wav`; `poll_status.meta.transform` carries
`strength/input_duration/input_sr/input_channels/target_samples`.

### `/generate/loop` body

Same fields as `/generate`, plus:

| field  | default | notes |
|--------|---------|-------|
| `bars` | `8`     | one of `4, 8, 16, 32` |
| `bpm`  | —       | optional; otherwise parsed from the prompt (e.g. `"... 124 bpm"`) |

DAW passes BPM inside the prompt. Loop length is exact 4/4
(`seconds_per_bar = (60/bpm)*4`); we generate `SA3_LOOP_PAD_SECONDS` (default
2.0) extra and hard-trim to `round(loop_duration*sr)` samples, so the loop is
DAW-bar-exact even for ugly BPMs. No peak-normalize (clamp path, same as
`/generate`). Response + `poll_status.meta.loop` carry
`bpm/bars/seconds_per_bar/loop_duration/gen_duration/target_samples`. Output
saved as `sa3loop_<sid>_<seed>.wav`.

### `/generate` body

| field           | default       | notes |
|-----------------|---------------|-------|
| `prompt`        | — (required)  | |
| `negative_prompt`| `low quality`| |
| `duration`      | `30`          | seconds, max 300 |
| `steps`         | `8`           | ARC default |
| `cfg_scale`     | `1.0`         | ARC runs at 1.0; APG/cfg-rescale inert there, not exposed |
| `shift`         | `default`     | `default` (model's own) \| `none` \| `logsnr` \| `flux` \| `full` |
| `sampler_type`  | `pingpong`    | |
| `seed`          | `-1`          | -1 → random; resolved server-side and returned for reproducibility |
| `loras`         | —             | list: `[{name,strength,interval_min,interval_max,layer_filter?}]`; unlisted registry LoRAs get strength 0 |
| `lora`          | `default`     | legacy, used only if `loras` absent: `default`\|`none` |
| `lora_strength` | `1.0`         | legacy, pairs with `lora` |

`interval_min`/`interval_max` sigma-gate that LoRA inside the DiT forward
(high `0.5–1.0` ≈ structure/chords, low `0.0–0.5` ≈ timbre) — same semantics
as the validated gradio multi-LoRA blend.

### Other endpoints

| Method | Path      | Purpose |
|--------|-----------|---------|
| GET    | `/loras`  | registry: `index → name`, default LoRA |
| POST   | `/reload` | rescan `~/sa3/loras/`, rebuild registry (409 if a generation is in flight) |
| POST   | `/unload` | free GPU memory in-process (409 if generating); reports `freed_mb`, before/after `cuda_mem` |
| POST   | `/load`   | (re)load the model warm; reports cold-load `load_seconds` |

### Model lifecycle (VRAM management)

`MANAGE_MODEL_LIFECYCLE` env (default **`off`** — model stays warm). Set
`on`/`true`/`1` and the worker frees GPU memory after every generation
(in-process: drop pipeline → 2× `gc.collect()` → `empty_cache()` →
`synchronize()`, the proven ace-step `/v1/unload` pattern), lazily cold-loading
on the next request. The unload runs *before* releasing the concurrency=1 `sa3`
queue lane, so no queued request can rebuild mid-free.

Benchmark without flipping the env using the manual `/unload` + `/load`
endpoints. `/health` reports `model_resident`, `manage_lifecycle`,
`last_load_seconds`, `cuda_mem`; `poll_status.meta` reports `model_load_time`
(cold-load seconds incurred by that request, `0` if it was warm). Measure
**both** reload latency and that `cuda_mem.allocated_mb`/`free_mb` actually
move — on the GB10's unified memory, "fast" doesn't guarantee "reclaimed".

## Build & run

```bash
./sync_engine.sh                                  # vendor current-main source
cp /path/to/blessed.ckpt loras/kev.ckpt
docker build -f Dockerfile.spark -t sa3-api:spark .
# brought up via gary-backend-spark/docker-compose.yml (service: sa3)
```

`engine/` and `loras/*.ckpt` are gitignored — regenerate via `sync_engine.sh`
and copy the blessed checkpoint per release. `engine/SOURCE_COMMIT` pins the
vendored commit.
