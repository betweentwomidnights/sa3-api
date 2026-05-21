#!/usr/bin/env bash
# Vendor the stable_audio_3 inference package from ~/stable-audio-3 (current main)
# into ./engine so the API image is self-contained and reproducible — no
# cross-repo bind mount at runtime. Re-run this whenever ~/stable-audio-3 main
# advances and you want the served inference code to track it, then rebuild.
set -euo pipefail

SRC="${1:-$HOME/stable-audio-3}"
DEST="$(cd "$(dirname "$0")" && pwd)/engine"

if [ ! -d "$SRC/stable_audio_3" ]; then
  echo "error: $SRC/stable_audio_3 not found" >&2
  exit 1
fi

echo "Vendoring stable_audio_3 from: $SRC"
echo "  commit: $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo '?') ($(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?'))"

rm -rf "$DEST/stable_audio_3"
mkdir -p "$DEST"
rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$SRC/stable_audio_3/" "$DEST/stable_audio_3/"

git -C "$SRC" rev-parse HEAD > "$DEST/SOURCE_COMMIT" 2>/dev/null || true
echo "Vendored -> $DEST/stable_audio_3 (pinned commit in engine/SOURCE_COMMIT)"
