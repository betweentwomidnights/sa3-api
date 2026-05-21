# SA3 API — JUCE Frontend Integration Handoff

Audience: the agent wiring the **Stable Audio 3 tab** in the gary4local VST.
You also have `api.py` — this doc is the contract + the client-side
responsibilities + the gotchas. Read this first, `api.py` for specifics.

## TL;DR — it's the foundation-1 poll contract

`/sa3/poll_status/<id>` returns **the exact JSON shape gary4local already
uses for foundation-1** (`success`, `generation_in_progress`,
`transform_in_progress`, `progress` 0–100, `status`, `queue_status`,
`audio_data` (base64 WAV) on completion, `error` on failure). **Reuse the
foundation poller path** — submit → get `session_id` → poll until
`status == "completed"` → base64-decode `audio_data`. Everything below is
just which endpoint to POST and what body to send.

## Base URL

```
https://g4l.thecollabagepatch.com/sa3
```

Routed through the same Cloudflare tunnel → caddy as `/foundation/*`,
`/carey/*`. No auth (single-user beta; the backend GPU queue is the gate).
All generation is **async**: POST returns a `session_id` immediately;
poll `GET /sa3/poll_status/<session_id>`.

## Generation endpoints

All accept JSON. Shared optional fields (defaults in parens):

| field | default | notes |
|---|---|---|
| `prompt` | (required) | see "Prompt responsibilities" below |
| `negative_prompt` | `"low quality"` | |
| `steps` | `8` | ARC default; don't expose unless "advanced" |
| `cfg_scale` | `1.0` | ARC runs at 1.0; leave it |
| `shift` | `"default"` | `default`\|`none`\|`logsnr`\|`flux`\|`full` (materially changes output) |
| `sampler_type` | `"pingpong"` | leave default |
| `seed` | `-1` | -1 → server picks; **the response always echoes the concrete seed — store it to reproduce a take** |
| `loras` | — | `[{ "name", "strength", "interval_min", "interval_max" }]` (see LoRA section) |
| `lora` / `lora_strength` | `"default"` / `1.0` | legacy single-LoRA; ignore if you send `loras` |
| `latent_rescale` | server default | loudness — **don't expose**; backend-tuned. Overridable per-request for A/B only |
| `latent_shift` | server default | loudness — **don't expose**; backend-tuned |
| `peak_normalize_db` | server default | legacy ace lever, **off by default** (pins dynamics); `"off"` disables. **Don't expose** |
| `limiter_ceiling_db` | server default (≤0) | audio-domain true-peak soft limiter — the active anti-clip net; `"off"` disables. **Don't expose** |

> Loudness: the blessed LoRA runs hot and clips long generations near the end.
> The server applies a latent rescale + peak-normalize ceiling by default, so
> output is clean out of the box. These three fields exist only so the backend
> can retune by ear without a redeploy — **leave them out of the UI**; the
> response `meta.loudness` echoes what was applied.

### `POST /sa3/generate` — text → audio
Extra: `duration` (float secs, default 30, max 300).
Returns `{success, session_id, seed, prompt, duration}`.

### `POST /sa3/generate/loop` — bar-aligned loop
**No `duration`.** Extra:
- `bars` — `4`\|`8`\|`16`\|`32` (default 8)
- `bpm` — optional; normally **parsed from the prompt** (the VST already
  writes "… 124 bpm …" — see below). Pass `bpm` only as an override.

Output is sample-exact bar length. Returns `{… bpm, bars,
seconds_per_bar, loop_duration, gen_duration, target_samples}`.

### `POST /sa3/transform` — style transfer of recorded audio
**No `duration` — output length == input length.** Extra:
- `audio_data` — base64 WAV recorded in the DAW (required)
- `strength` — `0.01`–`1.0` (default `0.9`). 0.01 ≈ preserve input,
  1.0 ≈ full transform. **NONLINEAR — see Known Quirks.**

Output is the **exact same length & channel count** as the input
(sample-exact round-trip; drops back into the DAW lined up).

### `POST /sa3/continue` — continue recorded audio
**No `duration`.** Extra:
- `audio_data` — base64 WAV source (required)
- `continuation_seconds` — float, default `8.0`
- `continuation_mode` — `"inpaint"` (default). `"latent_prefix"` is
  reserved and currently returns 400 — **don't expose it yet.**
- `continuation_tail_pad` — float seconds, **default `6.0`**. **Expose as an
  advanced slider** (range `0`–`60`, default `6`). Controls how the
  continuation *ends*:
  - `0` ≈ the model composes a full ending right at the cut (may sit quiet
    just before the cut)
  - `~6` (default) ≈ natural wind-down + a little tail before the cut
    (gradio-like; "you hear it ending")
  - `~20+` ≈ seamless — no taper, hard cut at full energy (best for chaining
    into another continue / endless playback)

  Suggested tooltip: *"How much the continuation winds down before the cut.
  Low = it sounds like an ending; high = seamless, cuts at full energy."*

Keeps the source, generates new audio after it. Output length =
source + continuation (sample-exact). The kept region is re-encoded
(musically continuous, not a bit-identical join — don't promise the user a
sample-perfect splice of their original).

> Background: early builds faded out ~the last 5s (a mask/conditioning bug).
> Fixed — the model is now told the piece runs `tail_pad` seconds past the
> cut, so the slider is purely the musical "ending vs seamless" choice.

## Prompt responsibilities (IMPORTANT — client-side)

The model is **256-token capped** and was LoRA-trained on
`"{genre}, {bpm} bpm, {key}"`. The **VST appends BPM and key/scale** to the
prompt itself (as it already does for stable-audio-api). So:

- The dice/`/prompts` pool entries are **genre/vibe only — no bpm/key.**
- Before sending, the VST composes the final prompt =
  `"<user or dice prompt>, <hostBPM> bpm, <key> <scale>"`.
- For `/sa3/generate/loop`, the BPM in that composed prompt is what the
  server parses for bar math (regex `(\d+(?:\.\d+)?)\s*bpm`,
  case-insensitive, matches mid-sentence). Keep writing it the way you do
  for stable-audio-api and it just works.

## LoRA model (the SA3 tab's signature feature)

- `GET /sa3/loras` → registry: `{loras:[{index,name}], default_lora}`.
  Build the LoRA list/sliders from this.
- Per request, send `loras` as a list — one entry per LoRA whose slider
  is up:
  ```json
  "loras": [
    {"name":"kev","strength":1.0,"interval_min":0.0,"interval_max":1.0}
  ]
  ```
  Omit a LoRA entirely (or strength implicitly 0) and it contributes
  nothing. `strength` 0–~2. `interval_min/max` (0–1, sigma-gated):
  high `0.5–1.0` ≈ structure/chords, low `0.0–0.5` ≈ timbre. Default a
  single LoRA to `strength 1.0, interval 0.0–1.0`.
- **Two LoRAs at once is supported** — send two entries with their slider
  strengths; they blend.

### Dice button — `GET /sa3/prompts`

`GET /sa3/prompts?lora=<name>` (repeatable / comma: `?lora=a,b` or
`?lora=a&lora=b` — **send every LoRA whose slider is up**). Returns:
```json
{ "success": true, "loras": [...], "missing_loras": [...],
  "available_loras": [...],
  "prompts": { "version":1,
    "dice": { "generic":[...], "instrumental":[...], "drums":[...] },
    "source": {...} } }
```
- No `lora` → generic pool.
- One/more LoRAs → buckets those LoRAs define become the **deduped union**
  across them (so a roll lands in either LoRA's training distribution);
  other buckets stay generic.
- Dice UX: roll = pick a random entry from the relevant bucket, then the
  VST appends bpm/key as above.
- These pools are **server-side live-editable** (no plugin reship needed to
  change them) — don't hardcode prompts in the VST; always fetch.

## Polling contract (reuse foundation code)

`GET /sa3/poll_status/<session_id>` →
```json
{ "success": true,
  "status": "queued|generating|encoding|completed|failed",
  "generation_in_progress": bool, "transform_in_progress": bool,
  "progress": 0-100,
  "queue_status": { ...same shape as foundation/g4lwebsockets... },
  // when completed:
  "audio_data": "<base64 WAV>", "meta": { ... },
  // when failed:
  "success": false, "error": "..." }
```
`meta` carries the resolved params incl. `seed`, and per-mode blocks
(`loop`, `transform`, `continue`) with the exact durations/sample counts —
useful for placing audio on the timeline. Output is 44.1 kHz stereo WAV.

Error shape for bad requests: HTTP 4xx + `{"success":false,"error":"..."}`
(or `{"success":false,"errors":[...]}` for validation). 503
`"loading model — warming up"` if you hit it before warmup (rare; the model
stays warm by default).

## Known quirks / don't-do

- **`transform` `strength` is perceptually nonlinear** — ~0.2 and ~0.7 sound
  similar, the action is ~0.7–1.0, 1.0 ≈ no resemblance. A perceptual
  remap is planned server-side; for now, if you expose the slider, bias its
  travel toward the top end or label it clearly. Flag to backend before
  shipping that slider.
- **Don't expose `continuation_mode=latent_prefix`** — not implemented (400).
- **Seed**: always read it back from the response/`meta` and keep it with
  the take so "regenerate / variation" can reuse or perturb it.
- **Lengths are sample-exact** for loop/transform/continue — you can place
  results on the DAW grid without re-trimming.
- Don't send bpm/key inside dice prompts — the VST adds them; doubling
  confuses the model.

## Health

`GET /sa3/health` → model/lora/cuda state. `GET /sa3/ready` → 200 when warm.
`/sa3/loras` for the registry. (Admin: `/sa3/reload`, `/sa3/unload`,
`/sa3/load` exist but are backend ops — not for the VST.)
