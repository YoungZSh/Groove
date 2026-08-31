#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${VISION_OPD_RUNTIME:-$PROJECT_ROOT/TMP/runtime/Vision-OPD}"
REPOSITORY="${VISION_OPD_REPOSITORY:-https://github.com/VisionOPD/Vision-OPD.git}"
COMMIT="${VISION_OPD_COMMIT:-c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471}"
PATCH_FILE="$PROJECT_ROOT/patches/vision_opd_visual_seed.patch"

if [[ -e "$TARGET" ]]; then
  echo "Runtime already exists: $TARGET"
  echo "Remove or relocate that exact directory yourself before rebuilding it."
  exit 0
fi

mkdir -p "$(dirname "$TARGET")"
git init "$TARGET"
git -C "$TARGET" remote add origin "$REPOSITORY"
git -C "$TARGET" fetch --depth 1 origin "$COMMIT"
git -C "$TARGET" checkout --detach FETCH_HEAD
git -C "$TARGET" apply "$PATCH_FILE"
echo "Prepared patched Vision-OPD runtime at $TARGET"
