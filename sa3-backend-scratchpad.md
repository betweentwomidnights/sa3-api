# SA3 Backend Scratchpad

Date: 2026-05-19 (last updated 2026-05-20)

## CURRENT SHIPPED — inference contract as of 2026-05-20

- **LoRAs (registry name-driven):** `kev` (default, blessed patch baseline — runs hot, +75% head→tail latent drift) + `koan` (sa3_training neutral pre-encode, step-3500 — drift killed). Loras dir bind-mounted in compose (`${HOME}/sa3/loras:/app/loras`), drop ckpt + recreate (or future `/reload`), no rebuild. Prompts pools `kev.json` (27 genres) + `koan.json` (12) live-mounted at `${HOME}/sa3/prompts`.
- **`/continue` tail:** `regen_past` mode default; `continuation_tail_pad` request slider 0–60s, default 6 (gradio-like natural wind-down + tail). 0 ≈ ending-at-cut, ≥20 ≈ seamless.
- **Loudness chain (mastering pattern, user ear-approved):** **peak-norm `+2.0 dB`** (gentle pre-scale, gain ≈0.755× of raw peak) → **soft-knee tanh limiter ceil `−0.3 dB` knee 0.8** → int16 clamp (no-op now). On koan @1.0 / 120s / seed 4242: combo body only 2.4 dB below raw, limiter touches just **0.21%** of samples (vs 8.23% limiter-only / vs full peak-norm’s 4.7 dB body drop), zero hard clip. Code change in this session: positive `peak_normalize_db` targets allowed (the chain needs target > limiter ceiling); positive limiter ceiling still coerced to None.
- **Per-request overrides** (no redeploy): `latent_rescale`, `latent_shift`, `latent_target_std`, `peak_normalize_db`, `limiter_ceiling_db`, `continuation_tail_pad`. `meta.loudness` echoes what was applied. **Backend-only — NOT in JUCE UI** per handoff.
- **Open inference items:** `latent_shift` (additive, before decode) implemented but never empirically tested — pull next session; long-continuation `sample_size` 120s cap is FIXED (lifted; verified 150s gen returns full samples).

## RESOLVED 2026-05-19 — `/continue` early-fade (Plan item 1)

Root cause confirmed by instrumented A/B (21s source → 100s total, seed 77777, same payload across modes; 5s-window RMS envelope on host):

- **legacy** (`duration=total+0.5`, `mask_end=total`): body −6.5 dB to 90s, then last 5s **−39.5 dB RMS / pk −18.6** (33 dB cliff into near-silence).
- **exact** (`duration=total`, `mask_end=total`): last 5s **−34.6 / pk −16.6** — barely helped. ⇒ the silent kept-island at `[total,total+0.5]` is the MINOR cause.
- **regen_past** (`mask_end=duration`, `seconds_total` pushed past trim, trim after) pad=6: **−22.5 / pk −7.3**; pad=20: **−3.8 / pk −0.0** (== body level, full-scale peaks — fade eliminated).

**Dominant driver = `seconds_total` conditioning**: the model composes an anticipatory wind-down wherever it thinks the piece ends. Fix = make it think the piece runs past the trim. Pad must exceed the model's end-anticipation horizon (>6s; 20s clean).

**Shipped:** env-toggled `SA3_CONTINUE_TAIL_MODE` (legacy|exact|regen_past) + `SA3_CONTINUE_TAIL_PAD`; code+compose default **regen_past / pad=20**. Worker logs tail5s pre/post-trim RMS/peak + `padded_short`; `poll_status.meta.continue` echoes `tail_mode/gen_duration/diag`. Compose (`gary-backend-spark`) pins it explicitly; live container already serving the fix. Cost: generates `total+20`(+pipeline's internal 6) then trims — ~20% extra compute at 8 steps (~+4s wall), acceptable.

**CAVEAT (follow-up, not blocking):** pipeline default `sample_size=5292032` = exactly 120s @44100; `_adapt_sample_size` clamps `audio_sample_size` to it. With pad=20, `total ≳ 94s` → requested gen (`total+20+6`) clamps to 120s → continuation truncated → worker zero-pads short → trailing silence returns (different failure than the fade). Long continuations (>~94s total) need either a larger `sample_size` passed through, or pad auto-scaled down near the cap. `meta.continue.diag.padded_short` will flag when this fires. Not hit in the 100s repro (126→clamp 120 ≥ target 100, no pad).

## Loudness FINDINGS 2026-05-19 — hotness origin LOCATED (Plan item 2 step a)

Env-gated `SA3_LATENT_DIAG` added (return_latents → log latent min/max/std + head/tail std, then decode + log waveform pk/rms whole/first5s/last5s/clip-frac). Results, 120s, seed 4242, same prompt:

| 120s seed4242 | latent std | absmax | head→tail std | decoded pk | clip frac |
|---|---|---|---|---|---|
| base `loras:[]` | 0.90 | 6.40 | 0.85→0.97 (+14%) | +1.47 dB | **0.000%** |
| default LoRA @1.0 | 1.33 | 9.48 | 0.94→1.65 (+75%) | +5.12 dB | **1.75%** |
| LoRA @1.0, 12s | 1.04 | 6.95 | 1.40→0.62 | +2.38 dB | 0.034% |

**Verdict:** hotness is born in the **LoRA's latents**, not the decoder (decoder faithful; base model clean at 2min). Blessed `kev` (formerly `sa3_patch_baseline`) inflates latent energy ~48% AND adds a time-drift (latent std +75% head→tail vs base +14%) → long gens clip hardest near the end (matches user's "2-min end clipped"). Scales with duration (12s 0.03% → 120s 1.75% clipped). Same family as v8 loudness regression but on the SHIPPED LoRA ⇒ must fix post-hoc in API (LoRA retrain = closed scope). Constant `latent_rescale ≈ 0.90/1.33 ≈ 0.68` pulls LoRA latent std back to base; won't fully kill the drift (tail stays relatively hot) → pair with post-decode peak-normalize (ace lever 2) as guaranteed anti-clip. Lever choice = next decision.

## Latent-scale colors timbre → two-stage SHIPPED 2026-05-19

User ear: heavy latent rescale may hurt quality. Tested decisively: matched clean 0.45 vs 0.60 latent-scale decodes (same seed4242) by best pure gain → **15.85% non-gain residual** (0% = clean volume) + spectral tilt concentrated **1.3–5kHz (−1.2..−1.6dB)** = midrange/presence coloration (rectangular STFT, treat per-band as directional; time-domain residual is exact). Confirms: latent scale ≠ clean gain (D(αz)≠αD(z)); heavier factor = more coloration. ⇒ architecture fix (user chose "gentle latent + audio soft-limiter"): keep adaptive latent factor near unity (raise target_std so factor stays high) + do anti-clip in AUDIO domain (lossless gain) via a C1-smooth tanh **soft-knee true-peak limiter** (identity below knee K=ceil*KNEE, asymptotes to ceiling C; only over-K samples touched). New env `SA3_LIMITER_CEILING_DB`/`SA3_LIMITER_KNEE` + per-request `limiter_ceiling_db` (`_resolve_db` shared w/ peak-norm); `meta.loudness` echo; limited% logged. **Prod env: regen_past/pad6 · RESCALE 1.0 · TARGET_STD 1.05 · ADAPT_MIN 0.7 · LIMITER -0.3dB knee0.8 · PEAK off · DIAG on.** Verified 120s seed4242: BASE factor **1.000** (zero coloration, RMS −13.82 == raw, limiter 0.0072%/766 samples — now a soft ceiling vs old hard clamp, strictly better, not quiet); LoRA factor **0.787** (gentler than 0.674 → less mid coloration), limiter 2.72%, final pk −0.30dB, **hard clip 1.75%→0.037%→0.000%**. Tradeoff dial (per-request, no redeploy): higher target_std = gentler limiter but more latent coloration; lower limiter_ceiling_db = more headroom. User ear-checking in DAW; watch LoRA-tail limiter squash (2.7%). Open: drift-aware still only true intro/outro rebalancer (deferred); DIAG off for prod; 120s sample_size cap for long continues.

## Adaptive rescale SHIPPED 2026-05-19 — production switched (user request)

User wanted "rescale-only" to eyeball base-vs-LoRA waveforms in Ableton, with the explicit goal: tame hot LoRA WITHOUT making base model / continuation source quiet. Flagged: a *constant* rescale fails that (attenuates clean base equally). Built **adaptive rescale** instead: `factor = clamp(target_std / latent.std(), MIN, MAX=1.0)`, MAX=1.0 ⇒ only attenuates, never amplifies (can't cause clipping). Env `SA3_LATENT_TARGET_STD` (set ⇒ supersedes constant LATENT_RESCALE), `SA3_LATENT_ADAPT_MIN/MAX`; per-request `latent_target_std`; logged `adaptive rescale latent_std=.. target=.. -> factor=..` + `meta.loudness`. Verified 120s seed4242 target_std=0.9: base std0.903→factor **0.997 (untouched, pk1.18, 0% clip — identical to no-fix base)**; LoRA std1.335→factor **0.674** (pk1.25, clip **1.75%→0.037%**). Residual LoRA 0.037% (tail peakier than std implies — same drift; not a hard-zero guarantee like peak-norm — deliberate trade: natural dynamics > pinned ceiling). **Production env now: regen_past/pad6 · LATENT_RESCALE=1.0 · LATENT_TARGET_STD=0.9 · PEAK_NORMALIZE_DB= (off) · DIAG=1.** Escape hatches per-request (no redeploy): latent_target_std 0.8/1.0, peak_normalize_db -1.0 to A/B old guaranteed-ceiling. User eyeballing waveforms in DAW next; human-in-the-loop loudness call is the intended workflow.

## Params exposed 2026-05-19 (user request)

- **`continuation_tail_pad`** now a per-request field on `/continue` (float secs, default = env `SA3_CONTINUE_TAIL_PAD`, **now 6**, capped `SA3_CONTINUE_TAIL_PAD_MAX`=60). It's the musical "ending vs seamless" slider: 0≈full composed ending at cut, ~6 gradio-like wind-down+tail (default — user wants "hear the end" by default), ~20+ seamless hard-cut. Echoed in response + `meta.continue.tail_pad`. Frontend: expose as advanced slider 0–60 def 6 + tooltip (handoff doc updated). No redeploy to retune.
- **`latent_rescale`/`latent_shift`/`peak_normalize_db`** also per-request overrides of the env loudness defaults (all gen endpoints, via `_build_params`; `_opt_float`/`_resolve_peak_db`; `peak_normalize_db:"off"` disables). Echoed in `meta.loudness`. NOT for UI — backend ear-tuning without redeploy; handoff says don't expose.
- **Empirically confirmed** the flagged interaction: /generate rescale=0.85 vs 0.6 (peak-norm −1 on) → DECODED peak byte-identical (−1.00dB/0.8911), RMS Δ~0.6dB only. ⇒ constant latent_rescale is ~fully cancelled by peak-normalize (only VAE nonlinearity remains); it does NOT rebalance intro/outro. Real balance levers = lower the norm ceiling (headroom, not balance) or drift-aware normalization (Plan option D, deferred pending user ear-verdict). User accepts quiet-intro (dataset/music teaches big-ending climax; "quiet > clipping").
- Compose pins regen_past / TAIL_PAD=6 / DIAG=1 / RESCALE=0.6 / NORM=-1.0. Code defaults stay opt-in (rescale 1.0/shift 0/norm off; tail mode regen_past, pad 6).

## Loudness FIX 2026-05-19 — two-stage shipped, clipping eliminated

Implemented ace's two-stage design env-gated: `SA3_LATENT_RESCALE` (def 1.0) / `SA3_LATENT_SHIFT` (def 0.0) applied to latents pre-decode (forces return_latents+manual-decode path since pipeline.generate clamps internally at pipeline.py:316 — clip damage is otherwise pre-baked), then `SA3_PEAK_NORMALIZE_DB` (empty=off; <=0 only) post-decode peak-normalize before the int16 clamp. Tested rescale=0.6 + norm=-1.0, 120s seed4242:

| 120s seed4242 | peak | clip | first5s rms | last5s rms |
|---|---|---|---|---|
| LoRA un-fixed | +5.12 dB | 1.75% | −16.7 | −11.8 |
| LoRA fixed | −1.00 dB | **0.000%** | −22.3 | −18.0 |

Clipping ELIMINATED (1.75%→0), peak exactly at target, 12s also clean. **Residual (predicted):** constant rescale can't undo the LoRA latent time-drift → ~4.3 dB head→tail imbalance survives (no longer distorts); global-peak normalize keys off the hot tail so the intro is pushed low (first5s pk −8.6 dB). Perceptual call (build vs loop) = user ear-check. If unacceptable → drift-aware latent normalization (Plan option D) is the motivated follow-up. Compose pins rescale=0.6/norm=-1.0/DIAG=1 (turn DIAG off for prod — though the fix forces the latent path anyway, so only logging overhead). Defaults in code still 1.0/0.0/off (opt-in) — compose is source of truth.

## Loudness recon 2026-05-19 — ace-step's exact levers (Plan item 2)

ace-step (`~/ace/ACE-Step-1.5`) tames hot output post-hoc, no retrain, two stages:
1. **Latent rescale/shift before VAE decode** — `acestep/core/generation/handler/generate_music_decode.py:89`: `pred_latents = pred_latents * latent_rescale + latent_shift` (defaults rescale 1.0 / shift 0.0; gradio shift slider ∈ [−0.2,0.2]). Reduces energy *before* decode ⇒ VAE never makes a clipped waveform. = user's latent-space hypothesis.
2. **Post-decode peak normalize**, default ON, target −1 dBFS — `acestep/audio_utils.py:normalize_audio`: `gain=10^(db/20)/peak; audio*=gain`, skip if peak<1e-6. Only when `normalization_db<=0`.

SA3 worker (`~/sa3/api.py`) currently does neither — only `audio.float().clamp(-1,1).mul(32767).int16` (hard clip @0dBFS). Continue A/B above shows pk −0.0 dB across all body windows = riding ceiling/clipping = the reported "hot". SA3 pipeline supports `return_latents=True` (pipeline.py:106/311) + `pretransform.decode()` (:508) ⇒ both instrumentation (latent-vs-decoded peak/RMS, short vs 2min) and the latent-rescale lever are mirror-able. Tie to v8 loudness regression / per-track-RMS history in [[stable-audio-3-beta-current-phase]]. Keep clamp path; new stage env-gated opt-in until A/B-by-ear proven (same method as the continue fix).

## Observations

- Outputs are coming back quite hot. This was noticeable in earlier curl tests too, and the end of a 2 minute `/sa3/generate` output sounded like it may have clipped.
- Frontend smoke tests passed for loops, 2 minute duration generation, fixed seed reuse, random seed echo/reuse, shift selection, and key/scale prompt appending.

## Backend Follow-Up

- Check whether SA3 output amplitude should be normalized, limited, or otherwise gain-managed before returning WAV data.
- If possible, inspect whether the hotness is already present in latent/pretransform output or only appears after decode/render.
- Compare short loop outputs against long duration outputs to see whether clipping risk increases near the end of longer generations.
- Align LoRA defaults with the frontend advanced UX: LoRA strengths should start at 0, and base-model/no-LoRA generation should be possible without the backend silently applying the default LoRA.
- Confirm `loras: []` truly produces base-model behavior with a controlled seed: run once with no LoRAs loaded, reload the LoRA registry, then run the same seed with `loras: []` and compare.
- Audit `/continue` against the private stable-audio-3 Gradio UI implementation. Current JUCE frontend now sends `continuation_seconds = requested_total_duration - source_duration`; the backend returns a WAV of the requested total length, but musical content often fades/stops before the final samples.
- Continue repro notes from DAW: same 21.1s source, target 100s total -> WAV is exactly 1:40, but generated content consistently stops around 1:36 with different seeds/LoRA strengths. Target 60s also seems to fade near the end. Target 120s stops around 1:57.5 with a small tail and about 2 seconds of silence.
- Current backend continue path to review: `continue_audio()` computes `total_duration = source_duration + continuation_seconds`, `target_samples = round(total_duration * sr)`, then sets `params["duration"] = total_duration + 0.5`, `mask_start_seconds = source_duration`, and `mask_end_seconds = total_duration`. Worker passes `duration`, `inpaint_audio`, `inpaint_mask_start_seconds`, and `inpaint_mask_end_seconds` into `pipe.generate()`, then trims/pads to `target_samples`.
- Specific continue questions: confirm mask polarity and units; confirm whether mask start/end are absolute seconds on the full generated timeline; check whether `mask_end_seconds` should be `total_duration` or the padded `duration`; test whether `/continue` should omit the `+0.5` pad or instead set mask end to the padded duration and trim afterward.
- Add backend diagnostics for continue tails: log if `have < target_samples` padding ever occurs, compute RMS/peak for the last 5 seconds before and after trimming, and include continue meta in the response/logs so DAW observations can be matched to source duration, requested continuation, total duration, mask start/end, generated duration, and target samples.
- If the early fade is intrinsic to the model, consider a backend tail strategy for automated continuous streams: request a small extra continuation region and trim/crop to the strongest musically active endpoint, or document that users should crop/continue from before the fade tail.
- Smart dice observation: when one or more LoRA sliders are above 0, some returned prompt rolls appear to include default-pool prompts mixed with LoRA-specific prompts. This may be a good feature rather than a bug, but confirm whether it is intentional. Current `/prompts` behavior starts from defaults, then only replaces buckets a selected LoRA defines, so any buckets not defined by that LoRA remain generic/default.
- Decide/document desired smart dice behavior: should LoRA-active dice roll across returned LoRA-defined + default fallback buckets, or should it constrain rolls to only buckets actually supplied by the selected LoRAs?

## Next-Session Plan (priority order)

**1. `/continue` early-fade + mask audit (concrete repro, likely a real bug, frontend-relevant).**
Leading hypothesis (verify, don't assume): `/continue` sets `duration = total_duration + 0.5` but `mask_end_seconds = total_duration`. The pipeline derives `audio_sample_size` from conditioning `seconds_total = duration`, and builds `inpaint_mask = ones; [start:end]=0` (0 = regenerate). So the region `[total_duration, total_duration+0.5]` ends up mask=1 ("keep") with no source content there → model doesn't generate it → tail fade/silence. The DAW repro (100s→stops ~1:36, 120s→~1:57.5 +~2s silence) fits a generated region that's effectively shorter than the requested grid.
Steps: (a) re-read `pipeline.py` inpaint section + `interface/diffusion_cond.py` inpaint wiring side by side, confirm mask polarity/units/absolute-timeline + the `effective_audio_len` mask-zeroing path; (b) fix — candidates: set `mask_end_seconds` to the *padded* `duration` (regenerate the whole post-source span, then trim), OR drop the `+0.5` and set `mask_end = duration = total_duration`; (c) add the diagnostics the scratchpad asks for (log if `have < target_samples` pad happened; RMS/peak of last 5s pre/post-trim; echo full continue meta) so model-intrinsic taper vs mask bug is separable; (d) re-run the 21.1s→60/100/120s repro.

**2. Loudness / hot outputs — post-hoc latent scale, NOT LoRA retrain (user's main interest).**
Goal: tame hot/clipping outputs (worse near end of long gens) without the abandoned pre-upstream custom training normalization. User's hypothesis: a latent-space shift/scale before decode (ace-step does this).
Steps: (a) instrument — peak/RMS of decoded WAV AND of latents pre-`pretransform.decode` (use `return_latents`), short vs 2-min, to locate where hotness originates (latent vs decode vs LoRA delta); (b) inspect `~/ace` for the latent normalization levers (VAE `scale_factor`/`shift_factor` / latent mean-std / pre-decode normalize) — find the concrete code; (c) prototype an env-gated post-hoc stage in the worker: latent shift/scale before decode and/or a gentle output peak-limit/gain; A/B by ear, same seed; (d) tie findings to existing loudness history (v8 regression / per-track-RMS) — see [[project_sa3_beta]]. Keep clamp path default; new stage opt-in until proven.

**3. Quick decisions/confirms (small).**
- `loras: []` must yield TRUE base model (no silent default LoRA). Confirm via scratchpad test (no-LoRA run → reload registry → same seed `loras:[]` compare). Likely change: omitted/empty `loras` ⇒ base model; default LoRA applied ONLY when explicitly named. Document for frontend.
- Smart-dice policy: decide LoRA-active = LoRA buckets + generic fallback (current) vs strict LoRA-only buckets. User leans "feature, keep". Just decide + document (trivial toggle if strict wanted).

Order: do **1** then **2** (both investigate→fix), **3** alongside as quick wins. Deferred still: `latent_prefix` continuation_mode, `transform` strength perceptual remap.

## Frontend Notes

- The JUCE SA3 tab now displays the backend-returned seed from random generations and can resubmit a specific seed via the `use seed` toggle.
- Advanced disclosure now matches Foundation's arrow treatment.
- When LoRA sliders are added, send an explicit `loras: []` array when all strengths are 0. The current API's legacy path applies the default LoRA when `loras` is omitted.
- Remote SA3 LoRA registry can be fetched/cached when the SA3 subtab becomes active. Localhost SA3 is disabled for now; when local support lands, reuse the SAOS/Carey-style refresh model because local LoRA files can change while the VST is open.
- Transform is now its own SA3 nested tab. It sends the selected recording/output WAV as `audio_data`, omits duration, and uses the shared seed/shift/LoRA advanced controls plus a transform-specific init noise strength.
- Continue is now its own SA3 nested tab. It sends selected recording/output WAV as `audio_data`, sets `continuation_seconds`, hard-codes `continuation_mode` to `inpaint`, and intentionally omits loop controls.
- Continue's visible duration slider is DAW-facing total output duration. The frontend subtracts the source audio duration before sending backend `continuation_seconds`, because the backend derives `total_duration = source_duration + continuation_seconds` and masks from source end to total end.
- SA3 prompt text is optional in the UI. Empty user prompts still submit because the frontend appends DAW BPM, enabling near-unconditional LoRA tests from the VST.
