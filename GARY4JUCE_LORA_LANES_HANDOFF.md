# gary4juce: LoRA automation lanes, one contract for remote and local SA3

Audience: the agent building the lanes UI in gary4juce. You also have the sa3.cpp repo. The
canonical references there are `docs/SERVER.md` ("LoRA automation lanes") and
`docs/C_ABI_V1.md` ("Automation lanes"). This note covers what gary4juce needs on top of them.

## The point

Remote SA3 (the PyTorch service on the Spark, `https://g4l.thecollabagepatch.com/sa3/*`) now
accepts the **same lane request** as local sa3.cpp. gary4juce needs one lanes UI and one request
builder, with only the transport different. Parity was checked step by step against sa3.cpp
(see "How close remote and local are").

## Feature detection (do this; old backends ignore the fields)

| backend | check | if absent |
| --- | --- | --- |
| remote | `GET /sa3/health` → `capabilities.lora_lanes == true` | old image: lanes ignored and the render has no automation |
| local sa3-server | `GET /health` → `capabilities.lora_lanes == true` (sa3.cpp PR #91) | older server: same, lanes ignored silently |
| local C ABI | `api->lane_init != NULL` | older lib: the lane fields are ignored |

Remote also reports `capabilities.lora_envelope_modes: ["velocity"]`, `max_lanes: 8` and
`max_lane_points: 256`.

**Status (2026-11-08):** remote is live (`sa3-api:spark-lanes`), and PR #91 is merged into
sa3.cpp main (v0.2.2). Still gate the UI on the capability, so older servers and libs degrade
cleanly.

## Request (HTTP JSON, identical for remote and sa3-server)

Works on `/generate`, `/generate/loop`, `/transform` and `/continue`, alongside all the
existing fields.

```json
{
  "prompt": "chiptune house groove 124bpm",
  "duration": 10, "seed": 42,
  "loras": [{"name": "kev", "strength": 0.5}],
  "lora_lanes": [
    {"loras": [{"name": "keygen", "strength": 0.8}],
     "envelope": [[0, 0], [4, 0], [6, 1.25], [10, 1]],
     "prompt": "demoscene chiptune, fast arpeggios"},
    {"loras": [{"name": "ratatat"}], "envelope": [[0, 1], [10, 0.3]]}
  ],
  "lane_average_until": 0.9,
  "lane_isolation": 1.0,
  "lane_isolation_until": 0.0
}
```

- `loras` (top level): **shared**. These run in every pass, at constant strength.
- `lora_lanes[]`: up to 8. Each lane holds `loras` (strength defaults to 1), an `envelope`
  (1–256 `[seconds, amount]` points, amount 0–100) and an optional `prompt`.
  - The envelope is linear between points and held flat past either end. `amount` multiplies the
    lane LoRA's `strength`, so with strength 1 the curve *is* the strength.
  - A lane with a `prompt` and no `loras` automates the prompt.
  - A lane with an all-zero curve, or with no LoRAs and no prompt of its own, is dropped.
- **Time 0** is the start of the output canvas. For `/continue` it's the start of the uploaded
  source audio, and for `/transform` the start of the input. Seconds past the requested duration
  (tail pad, loop pad) just hold the last value.
- `lora_envelope: [[sec, amount], ...]` with plain `loras` and no `lora_lanes` is the one-lane
  shorthand: every LoRA in `loras` follows that one curve.

### Send the blend settings explicitly

The C ABI's `request_init` defaults to **average 0.9 / isolation 1 / until 0** (the sa3.cpp
Automation tab uses the same). The HTTP defaults on **both** servers are **0 / 0 / 0.9**, which
means off. To make remote sound like local, always send the three fields from the plugin, using
whatever the user set or the C ABI defaults. In short:
- Averaging: for steps with sigma > `lane_average_until`, every lane plays at its curve's mean.
  The arrangement forms first, and then the curves shape it.
- Isolation: frames with a different lane mix attend to each other less, so sections keep their
  own character.

Both thresholds depend on the schedule. With `/transform`, sampling starts at the noise level,
so at strength 0.5 averaging at 0.9 never runs.

### Mapping to the C ABI (local, in-process)

| JSON | C ABI |
| --- | --- |
| `loras` | `request.adapters` (shared) |
| `lora_lanes[i].loras` | `lane.adapters` / `adapter_count` / `adapter_stride` |
| `lora_lanes[i].envelope` | `lane.envelope` = flat `float[]` of (sec, amount) pairs, `envelope_point_count` = pairs |
| `lora_lanes[i].prompt` | `lane.prompt` (NULL or "" = the request prompt) |
| `lane_*` | `request.lane_isolation`, `lane_isolation_until`, `lane_average_until` |
|  | `request.lanes`, `lane_count`, `lane_stride = sizeof(sa3_lane_v1)`. Call `api->lane_init(&lane)` first |

A lane adapter must report `adapter_info.supports_lanes`. Decoder adapters go in the request,
never in a lane.

## Response

The completed `/poll_status` carries `meta.lora_lanes` (remote), which echoes what actually ran:
the lanes after dropping, each lane's peak and point count, and the three blend values. It is
`null` when no automation ran.

## Things remote rejects (HTTP 400) that local accepts

Keep the UI inside these limits so a project behaves the same on both:
1. **The same LoRA both shared and in a lane, or twice in one lane.** Remote can't stack an
   adapter with itself. In the UI, a LoRA used in a lane should leave the shared list.
2. **`lora_envelope_mode: "layer"`.** Remote only does `"velocity"`, which is the default
   anyway, so don't send the field.
3. **A shared LoRA with a non-default `interval_min/max` or a `layer_filter` while lanes are
   on.** Drop those per-LoRA advanced settings when lanes are active.

Error bodies are `{"success": false, "error": "..."}`, worded like sa3-server's.

## LoRA names

Remote names come from `GET /sa3/loras`. Today they are `kev`, `keygen`, `koan`, `ratatat`
and `succession`, all DoRA (`dora-rows`), and all lane-capable. For local parity the same
adapters need to be converted to GGUF as `lora-<name>-f32.gguf`, so the name resolves on both
sides. (`.ckpt` → `tools/lora_ckpt_export.py` → `sa3-lora-convert --in <dir>/<name>`. `--in`
takes a path without the extension.)

## Cost (tell the user; size the queue ETA)

Each lane is one extra DiT pass per step (CFG doubles every pass). Remote, 10 s, 8 steps,
2 lanes + shared LoRA + isolation: ~7–8 s, against ~5 s with no lanes. Remote caches about
2.7 GB of weights per adapted pass, up to 12 GB. Past 4 adapted passes it gets slower, not
broken. Local sa3.cpp is in the same range (2 lanes, 10 s: ~10 s on the Spark).

## How close remote and local are

With the same seed, both sides draw the same noise (sa3.cpp `rng: "torch-cuda"`, 288 blocks =
the Spark). Teacher-forced on identical input, the lane math matches to the backend's arithmetic
floor (final latent rel 0.12–0.35% in the fp32 parity setup). **The audio is not byte-identical**:
remote runs fp16 and masks the headroom frames, so even plain renders differ by a few percent per
step and drift over the steps (about 10% in a 10 s latent; at 30 s, audio cos ~0.97). Expect the
same take, audibly very close, but not a null test. Don't build anything that assumes
remote == local sample for sample.

With a negative prompt and cfg > 1, local only matches remote after **sa3.cpp PR #91** (local
used to pad the negative prompt differently). At the default cfg 1 this doesn't matter.

## Pointers

- sa3.cpp: `docs/SERVER.md` (lanes, every field), `docs/C_ABI_V1.md` (Automation lanes),
  `docs/TORCH_PARITY.md` (parity numbers, lane table), and `web/lanes.js` (the reference editor
  UI: drawing curves over a take).
- Remote implementation: `~/sa3/lora_lanes.py`, `~/sa3/LORA_LANES.md`.
