# SA3 loudness & continuation env vars — defaults + what to expose

These are the server-side knobs that shape SA3's output loudness and the
`/continue` tail. On the Spark they live in `docker-compose.yml` as backend-only,
ear-tuned values. For gary4local: **use our values as defaults and expose the
loudness ones as an "advanced" panel** — `api.py` already accepts every one as a
**per-request JSON override** (omit → env default; the response echoes what was
actually applied in `meta.loudness`). So the UI just sends the field; no backend
change needed.

## Why these exist (the chain)

The blessed LoRA runs hot and clips long generations near the end. Two-stage fix:
**(1)** gently tame latent energy *before* decode, then **(2)** catch residual
peaks in the *audio* domain with a soft-knee true-peak limiter — audio-domain gain
is exactly linear, so it doesn't color timbre the way latent scaling does. A
post-decode peak-normalize sits between them. Current chain (ear-approved):
**peak-norm +2.0 dB → soft-knee limiter −0.3 dB (knee 0.8)**, adaptive latent
rescale off. On a hot LoRA the limiter only touches a fraction of a percent of
samples, with zero hard clip; base model / continuation source pass untouched.

## Loudness knobs — expose as advanced (defaults shown)

| Env var (server default) | Request field (per-call) | Our value | What it does |
|---|---|---|---|
| `SA3_PEAK_NORMALIZE_DB` | `peak_normalize_db` | `2.0` | Pre-scale the decoded waveform so its peak ≈ this dBFS, then let the limiter shave. `off`/empty disables. (A *positive* target only makes sense paired with the limiter — alone it would hard-clip.) |
| `SA3_LIMITER_CEILING_DB` | `limiter_ceiling_db` | `-0.3` | True-peak soft-limiter ceiling in dBFS (must be ≤ 0). The anti-clip net. `off` disables; positive is coerced off. |
| `SA3_LATENT_RESCALE` | `latent_rescale` | `1.0` | Constant latent multiply before decode (`1.0` = off). |
| `SA3_LATENT_SHIFT` | `latent_shift` | `0.0` | Latent add before decode (`0.0` = off). |
| `SA3_LATENT_TARGET_STD` | `latent_target_std` | *(off)* | Adaptive: scale latents *toward* this global std instead of a constant factor (base ≈ untouched, hot LoRA attenuated). Supersedes `latent_rescale` when set; try ~`0.9`. |
| `SA3_CONTINUE_TAIL_PAD` | `continuation_tail_pad` | `6` | `/continue` only — the musical "ending vs seamless" dial (`0` = full composed ending at the cut, `~6` = natural wind-down, `~20+` = seamless). Range 0–60. Already a good advanced slider. |

Override semantics: send the field in the request JSON; omit or `""` → env default;
the dB fields also accept `"off"`/`"none"` to disable.

## Backend-only knobs — set as defaults, NOT per-request

| Env var | Our value | Notes |
|---|---|---|
| `SA3_LIMITER_KNEE` | `0.8` | Where the soft region starts (fraction of the ceiling; `1.0` = hard, lower = gentler/earlier). |
| `SA3_LATENT_ADAPT_MIN` | `0.9` | Floor for the adaptive factor — only matters when `latent_target_std` is set. |
| `SA3_LATENT_ADAPT_MAX` | `1.0` | Cap — `1.0` means adaptive can only attenuate, never amplify (so it can't *cause* clipping). |
| `SA3_CONTINUE_TAIL_MODE` | `regen_past` | `/continue` tail strategy (fixes an early-fade bug). Leave it. |
| `SA3_LATENT_DIAG` | `0` for prod | `1` logs latent/decoded peak·RMS·clip diagnostics. The Spark runs `1`; turn off in prod. |

## Production defaults (copy as the gary4local baseline)

```
SA3_PEAK_NORMALIZE_DB=2.0
SA3_LIMITER_CEILING_DB=-0.3
SA3_LIMITER_KNEE=0.8
SA3_LATENT_RESCALE=1.0
SA3_LATENT_SHIFT=0.0
SA3_LATENT_TARGET_STD=          # empty = off (adaptive disabled)
SA3_LATENT_ADAPT_MIN=0.9
SA3_LATENT_ADAPT_MAX=1.0
SA3_CONTINUE_TAIL_MODE=regen_past
SA3_CONTINUE_TAIL_PAD=6
SA3_LATENT_DIAG=0
```

> Note: the JUCE handoff (`SA3_API_HANDOFF.md`) says "don't expose" these — that
> guidance was for the Spark's backend-only ear-tuning. For gary4local we *do*
> want the loudness fields as an advanced panel, so this doc supersedes it there.
