#!/usr/bin/env python3
"""
Seed a per-LoRA dice pool from that LoRA's SA3 training captions.

SA3 LoRA training (pre_encode_dataset.py:caption_metadata_fn) uses the FULL
text of each `<clip>.txt` as the prompt. For the patch dataset those are
`"{genre}, {bpm} bpm, {key}"` (the `genre` field — the rich `caption:` prose
in ace/training/patch_staging was ACE-step's, never SA3's). So the in-
distribution dice pool for a LoRA is its genre leads, with the
`, <bpm> bpm, <key>` tail stripped (the VST appends bpm + key/scale itself).

This is a host-side seeding tool. It writes ~/sa3/prompts/<name>.json which
is mounted live into the container; you then hand-curate that file and it is
never silently overwritten (re-run with --force to regenerate from scratch).

Usage:
  ./build_lora_prompts.py --name kev \
      --captions-dir /home/kev/sa3_training
"""
import argparse
import json
import os
import re
import sys

# Strip everything from the first "<n> bpm" token onward (also drops the key
# that trails it). Case-insensitive; tolerates "120bpm" / "120 BPM".
_BPM_TAIL = re.compile(r"[,;]?\s*\d+(?:\.\d+)?\s*bpm.*$", re.IGNORECASE | re.DOTALL)


def genre_from_caption(text: str) -> str:
    g = _BPM_TAIL.sub("", text).strip()
    return g.strip(" ,;\t\r\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    help="LoRA registry name (file stem of the .ckpt)")
    ap.add_argument("--captions-dir", required=True,
                    help="Dir of <clip>.txt SA3 training captions")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "prompts"))
    ap.add_argument("--bucket", default="instrumental",
                    help="dice bucket (patch dataset is all-instrumental)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing curated file")
    args = ap.parse_args()

    if not os.path.isdir(args.captions_dir):
        sys.exit(f"captions-dir not found: {args.captions_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.json")
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"{out_path} exists — refusing to clobber curated prompts "
                 f"(use --force to regenerate)")

    seen, genres = set(), []
    txts = sorted(f for f in os.listdir(args.captions_dir) if f.endswith(".txt"))
    for fn in txts:
        with open(os.path.join(args.captions_dir, fn)) as f:
            raw = f.read().strip()
        g = genre_from_caption(raw)
        if not g:
            continue
        key = g.lower()
        if key not in seen:
            seen.add(key)
            genres.append(g)

    payload = {
        "version": 1,
        "source": {"lora": args.name, "captions_dir": args.captions_dir,
                   "files": len(txts), "unique_genres": len(genres)},
        "dice": {args.bucket: genres},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {out_path}: {len(genres)} unique genres from {len(txts)} "
          f"captions -> dice.{args.bucket}")
    print("  " + " | ".join(genres[:12]) + (" ..." if len(genres) > 12 else ""))


if __name__ == "__main__":
    main()
