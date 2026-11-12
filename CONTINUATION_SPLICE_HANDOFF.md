# `/continue` seamlessness — source splice + mask overlap (gary4local, PC + Mac)

Audience: the agent implementing continuation in gary4local's SA3 backend
(torch/CUDA on PC, MLX on Mac). Reference implementation: `api.py` in this repo
(`_splice_continuation_source` ~line 897, env block ~line 286; mask overlap
~line 284 and ~line 2104). Both features ship on our remote backend.

**Two separate levers, two different artifacts** — you need both to match the
remote backend's seam:

| Lever | Fixes | Default in `api.py` |
|---|---|---|
| **Source splice** (`SA3_CONTINUE_SPLICE_SOURCE`) | round-trip AE squeak on the kept audio | `1` — on |
| **Mask overlap** (`SA3_CONTINUE_MASK_OVERLAP`) | the "silent 8th note" rest at the downbeat | **`0.0` — OFF in code; our deploy sets `0.2`** |

> ⚠️ **If gary4local ported `api.py`'s code defaults, mask overlap is off.** That
> is the most likely reason a port sounds less seamless than our backend even
> with the splice working perfectly — the two fix different things and the
> deployed value lives in docker-compose, not in the source.

## What it is, in one paragraph

A continuation re-encodes the user's source audio through the autoencoder, so
the "kept" head of the output is the **VAE's re-render** of the input, not the
input. SAME downsamples 4096× (~10.8 latent frames/sec, one frame per ~93 ms of
stereo), so it cannot carry a snare or hat attack — it *resynthesizes* one,
slightly differently every pass. Measured over 6 round trips: added HF energy
climbs ~15 dB while lost HF stays flat (the AE **manufactures** high end), and
the damage concentrates on transients (transient-vs-sustain excess +4.2 → +12.0
dB). Chained continuations in the app run that loop N times, so the squeaks
compound. **Fix: after decode, paste the original source waveform back over the
kept region so it never accumulates a round trip.** The regenerated region is
untouched — that content only ever existed as model output.

## The two rules that make it sound seamless

1. **Splice BEFORE the loudness chain.** The pasted-in original and the
   generated tail must go through the *identical* peak-normalize + limiter
   stages, or you get two spliced takes at two levels instead of one song. In
   `api.py` the splice sits immediately after `pretransform.decode(...)` and
   before stage 2a (peak-normalize) / 2b (limiter). This is the delicate part —
   it's the only ordering constraint, and getting it wrong is audible.
2. **RMS gain-match the source to the model's re-render of the same region**
   before crossfading. Scalar gain only (never touches transients). Without it
   the seam has a level step, and whichever side is hotter drives the downstream
   peak-normalize. Clamp the factor (we use 0.25–4.0) so a near-silent source
   can't blow up.

## The algorithm

Inputs: decoded model output `audio` `[b, ch, T]` at model SR; the original
source waveform + its SR; `mask_start_seconds`.

```
mask_start <= 0                      -> no-op
resample source -> model SR
match channels (mixdown or duplicate) to audio's channel count
splice_end = min(round(mask_start * sr), source_len, audio_len)
xfade      = clamp(round(SPLICE_XFADE * sr), 0, splice_end // 2)
hard_end   = splice_end - xfade

if gain_match:
    gain = clamp(rms(audio[..., :splice_end]) / rms(src), 0.25, 4.0)   # rms>1e-6 guard
    src *= gain

out = audio.clone()
out[..., :hard_end]              = src[..., :hard_end]
out[..., hard_end:splice_end]    = src * cos(r*pi/2) + audio * sin(r*pi/2)   # r = linspace(0,1,xfade)
# out[..., splice_end:] untouched — pure model output
```

Notes that matter:

- **The source is head-aligned at sample 0.** SA3's `PadCrop(randomize=False)`
  writes the source at index 0 and zero-pads the tail, so the kept region is
  exactly `[0, mask_start)`. If your backend aligns differently, fix the offset
  here first — everything else assumes it.
- **Equal-power (cos/sin) crossfade, kept short (~30 ms).** The two signals are
  the same music (source vs its re-render): correlated low, decorrelated high.
  A long blend combs. 30 ms kills the click and is short enough that the comb is
  inaudible. Don't "improve" this by lengthening it.
- **The seam is not the musical hand-off.** It sits just *before* `mask_start`,
  inside the region where both signals are the same music. The musical
  transition is still at `mask_start`, handled by the model.
- **`mask_start` is usually < the source end.** With mask overlap on,
  `mask_start = source_duration - mask_overlap`. Splice to `mask_start`, **not**
  to the source end — the overlap slice is deliberately regenerated. See the
  mask-overlap section below.
- **Applies to every continuation mode**, `inpaint` and `latent_prefix` alike —
  it's post-decode and sampler-agnostic. On the Mac/MLX path it applies the same
  way: it's plain array math on the decoded waveform, no torch required.
- **Only `/continue`.** `/generate`, `/generate/loop`, `/transform` don't have a
  kept source region. (`/transform` is *supposed* to re-render everything.)
- Clone before writing so you don't mutate a tensor/array the caller still holds.

## Mask overlap — the other half of seamlessness

### The artifact it fixes

When the source clip ends **exactly on a bar line** (a clean 4-bar loop at 120
BPM = exactly 8.0 s), the model very often comes in a beat late: a short **silent
8th note**, sometimes a quarter, right at the hand-off. On a kick-driven loop it
reads as a *skipped downbeat*.

This is not a glue bug and not a level duck — the model is **composing a rest**.
A clip that ends on the bar looks like a completed phrase, so it does the
conventional thing and leaves the breath before the next downbeat. The giveaway
is that the gap is a clean musical subdivision and it *varies* run to run; a
technical artifact would be a constant sub-frame gap. SA3 continuation is
conditioning-based inpaint — the region after the source is seeded with zeros and
the model may compose anything that fits, including silence. Nothing forces
energy continuity across the seam. (RePaint-style engines like ace-step paste the
known audio back every denoise step, so they physically can't drop to silence at
the join.)

### The fix

**Pull the inpaint mask START back `overlap` seconds before the true end of the
source**, so the kept region ends *mid-phrase* instead of on the bar line:

```
mask_start = max(0, source_duration - mask_overlap)
mask_end   = unchanged   # still derives from source + continuation (+ tail_pad)
```

The last `overlap` seconds of the source are **regenerated rather than kept**.
That's the whole trick: the model no longer sees a cadence to resolve, so it
grooves straight through the downbeat.

**`mask_end` and `target_samples` must not change.** Output length still derives
from `source + continuation`, so the model's grid-aligned continuation lands
where it should — you are moving the *seam*, not the *timeline*. Getting this
wrong shifts everything by `overlap` and puts the new material off the grid,
which sounds worse than the rest you were fixing.

Clamp so you always keep real source to condition on:
`mask_overlap = min(mask_overlap, max(0, source_duration - 0.05))`, reject
negatives with a 400.

### How it composes with the splice

```
   0                          mask_start        source_end        mask_end
   |------------ kept ------------|--- overlap ---|--- new audio ---|
   |<-- original pasted back -->|xf|<---- model output, untouched ---->|
                                 ^                ^
                        30ms equal-power     the source's real
                        crossfade, both      end — now INSIDE the
                        sides same music     regenerated region
```

- The splice keeps `[0, mask_start)` — i.e. source **minus** the overlap. The two
  features stack cleanly: splice protects the kept head from round-trip damage,
  overlap moves the hand-off off the grid.
- The crossfade sits *just before* `mask_start`, still inside source-vs-re-render
  territory. Overlap doesn't change that.
- Consequence to be aware of: the user's final `overlap` seconds of recorded
  audio are **replaced by model output**. At 0.2 s that's inaudible and worth it.
  At 1 s+ you're rewriting real performance the user played — which is the reason
  for the upper bound below, and worth a sentence in the advanced-panel tooltip.

### Tuning

- **`0.2` is our shipped value** (≈ an 8th-to-quarter note at typical tempos).
  Start there; it's what the remote backend runs.
- **Too small (< ~0.1 s)** — doesn't clear the cadence cue, the rest comes back.
- **Too large (> ~0.5 s)** — you throw away real source content and give the
  model more room to drift away from the groove it was handed.
- Scale with tempo if you want to be clever: roughly one 8th note,
  `30 / BPM` seconds. Not required — anywhere off the grid works, precision
  isn't the point.
- `0.0` disables it (seam exactly at the source end) — keep that as the A/B.
- **Manual workaround, no code needed:** just don't end the capture on the bar
  line. Recording ~8.2 s of a 120 BPM 4-bar loop instead of 8.0 s has the same
  effect. Useful for explaining the behavior to users, and a good sanity check
  that you're chasing the right artifact.

## Configurability — mirror the loudness pattern

Same shape as the loudness knobs in `LOUDNESS_TUNING.md`: **env var default +
per-request JSON override, response echoes what was actually applied.** In our
`api.py` these are currently **env-only**; for gary4local, plumb the per-request
overrides too and put them in the advanced panel next to the loudness fields.

| Env var (default) | Suggested request field | Our value | What it does |
|---|---|---|---|
| `SA3_CONTINUE_SPLICE_SOURCE` | `splice_source` | `1` | Master on/off. `0` = old behavior (kept region is the AE re-render). Keep the off-switch — it's the one-line revert if a seam ever turns out audible. |
| `SA3_CONTINUE_SPLICE_XFADE` | `splice_xfade` | `0.03` | Crossfade length, seconds. Sane range ~0.005–0.10. Longer combs, shorter clicks. |
| `SA3_CONTINUE_SPLICE_GAIN_MATCH` | `splice_gain_match` | `1` | RMS-match the source to the re-render before blending. Leave on. |
| `SA3_CONTINUE_MASK_OVERLAP` | `continuation_mask_overlap` | **`0.2`** (code default `0.0`) | Seconds of source regenerated instead of kept, so the seam lands mid-phrase. Already per-request in our `api.py`. Sane range 0–0.5. |

Boolean env parsing: treat `0/false/no/off/""` as off (matches `api.py`).

Echo the applied values back in the response — we return `splice_source` /
`splice_xfade` on the `/continue` ack and the full continuation block in
`meta.continue`. Do the same so the UI can show what actually ran.

Baseline block for gary4local (add to the loudness defaults from
`LOUDNESS_TUNING.md`):

```
SA3_CONTINUE_SPLICE_SOURCE=1
SA3_CONTINUE_SPLICE_XFADE=0.03
SA3_CONTINUE_SPLICE_GAIN_MATCH=1
SA3_CONTINUE_MASK_OVERLAP=0.2      # NOT optional if you want our seam — code default is 0.0
```

Also carry over the tail knobs from `LOUDNESS_TUNING.md` — they shape the *end*
of the clip rather than the seam, but they're the other half of "sounds like our
backend": `SA3_CONTINUE_TAIL_MODE=regen_past`, `SA3_CONTINUE_TAIL_PAD=6`.

## How to verify it works

1. **Null test.** Continue a clip with splice on; the output head `[0,
   mask_start)` should be the input, sample-for-sample, modulo the gain factor
   and the loudness chain. Diff it against the input scaled by the reported
   gain — the error should be tiny outside the crossfade window.
2. **Chain test.** The real payoff. Run 4–6 chained continuations with splice
   off, then on. Off: hats/snares in the *early* part of the clip get
   progressively brighter and squeakier each pass. On: the head stays clean —
   only the newest tail is ever a first-generation render.
3. **Seam listen.** Solo around `mask_start` at high gain. You should hear no
   click, no level step, and no phasey "flanged" ~30 ms. A level step means the
   splice landed after the loudness chain, or gain-match is off.
4. Log a line per splice like ours:
   `continue splice: kept 0-8.000s xfade 30ms gain 1.024`. A gain far from 1.0
   is a red flag (channel/alignment mismatch), not a level problem.

## Failure modes to watch for

- **Splice placed after peak-normalize/limiter** → the head and tail sit at
  different levels; sounds like two takes edited together. Most likely bug.
- **Splicing to the source end instead of `mask_start`** when mask-overlap is
  non-zero → you overwrite the model's deliberately-regenerated overlap and the
  "silent 8th note" fix stops working.
- **Channel mismatch** (mono source into a stereo model output, or vice versa)
  → duplicate/mixdown explicitly; don't rely on broadcasting.
- **SR mismatch** → resample the source to the *model* SR before splicing, not
  after (the output is written at model SR).
- **Crossfade longer than the kept region** → clamp `xfade` to `splice_end // 2`.

## Triage: "it's still not as seamless as the remote backend"

Work down this list — the artifacts are distinguishable by ear.

1. **A short silence / skipped downbeat at the hand-off, length varies per run**
   → mask overlap is off or too small. Check the effective value (log it), not
   the code default. This is the #1 suspect for a fresh port.
2. **Bright, squeaky hats/snares that get worse the more you chain** → splice off
   or splicing the wrong region. Check the `kept 0-X.XXXs` log matches
   `mask_start`.
3. **Audible level step at the seam** → splice ran after the loudness chain, or
   gain-match is off.
4. **Click or a ~30 ms phasey smear** → crossfade length. Too short clicks, too
   long combs.
5. **The new material feels rhythmically off the grid** → `mask_end` /
   `target_samples` moved with `mask_start`. They must not.
6. **Clip fades out near the end** → not a seam issue: that's `tail_mode` /
   `tail_pad` (see `LOUDNESS_TUNING.md`), `regen_past` + `6`.

Log one line per continuation with the *effective* values — mode, mask_start,
mask_overlap, splice on/off, xfade, gain, tail_pad — and most of this triage
becomes a glance at the log instead of a listening test.
