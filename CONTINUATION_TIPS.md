# Continuation tips — avoiding the "silent 8th note" at the seam

## The symptom

When you continue a source clip whose end lands **exactly on a bar boundary**
(e.g. a clean 4-bar loop at 120 BPM = exactly 8.0s), the model very often comes
in a beat late: you hear a short **silent 8th note** (sometimes a quarter note)
right at the transition from your source audio into the model's continuation.
On a kick-driven loop it can read as a *skipped* downbeat.

## What's actually happening

This is **not** a glue/stitching bug and **not** a level "duck" — it's the model
*composing a rest*.

- A clip that ends exactly on the bar line looks to the model like a **completed
  musical phrase**, so it does the conventional thing and leaves the little
  "breath" before the next downbeat — that's your silent 8th/quarter note.
- The giveaway is the duration: it's a clean musical subdivision (8th, quarter)
  and it *varies* run to run. A technical artifact would be a constant, tiny,
  sub-frame gap. A varying musical note value is the model phrasing.
- SA3's continuation is **conditioning-based inpaint**: the region after the
  source is seeded with zeros and the model is free to compose whatever fits —
  including silence. Nothing forces energy continuity across the seam, so at a
  phrase boundary it inserts the rest. (RePaint-style engines like ace-step paste
  the known audio back every denoise step, so they *can't* drop to silence at the
  join — though they can still grab the wrong hit.)

## The fix (no code change needed)

**Don't end your source exactly on the bar line. Record a little past it.**

Example: instead of capturing exactly 8.0s of a 120 BPM 4-bar loop, capture
**~8.2s**. That extra fraction of a beat removes the "phrase complete" cue, so
the model just keeps grooving straight through — the boundary becomes inaudible.

Rules of thumb:
- Add roughly an **8th-to-quarter note** of audio past the bar (≈0.1–0.3s at
  typical tempos). You don't need to be precise — anywhere off the grid works.
- Ending **mid-phrase** in general (not just past the bar) avoids the rest.
- Because of how the VST works you can just **re-record/iterate** — nudge the
  capture endpoint a touch and the duck disappears.

## Why this works

You're moving the model's hand-off point off the metrical phrase boundary. With
a "messy" (off-grid) endpoint there's no cadence for the model to anticipate, so
it continues the texture directly instead of resting before an imagined downbeat.

## The automatic fix — `SA3_CONTINUE_MASK_OVERLAP` (shipped)

Step one of the robust version is now implemented and on by default in
docker-compose (`SA3_CONTINUE_MASK_OVERLAP=0.2`; per-request override
`continuation_mask_overlap`, seconds). It pulls the inpaint mask **start** back
by that many seconds before the true source end, so the kept region ends
mid-phrase instead of on the bar line. No cadence cue at the seam → the model
grooves straight through the downbeat instead of composing a rest. `mask_end` /
output length are unchanged (still `source + continuation`), so the model's
grid-aligned continuation still lands where it should. 0.0 disables it (the old
seam-exactly-at-source-end behavior). ~0.2s ≈ an 8th-to-quarter note; retune by
recreating the container (`docker compose up -d sa3`, no rebuild).

Caveat: the overlap slice `[source-overlap, source]` is *regenerated*, not kept —
so the very tail of your real source is replaced by model audio there. It's
mid-bar content the model recreates convincingly, but it's not bit-exact source.

## Note for future us — the crossfade (still pending)

If `MASK_OVERLAP` alone still leaves an occasional gap, the fully seamless
version adds an **equal-power crossfade**: keep the real source bit-exact through
the overlap and crossfade it against the model's regenerated overlap. That hides
any weak onset *inside* the crossfade and keeps the real source intact — same
continuity RePaint gets, on top of our existing inpaint, with no sampler change.
Not implemented yet; `MASK_OVERLAP` is the lever to confirm whether we need it.
