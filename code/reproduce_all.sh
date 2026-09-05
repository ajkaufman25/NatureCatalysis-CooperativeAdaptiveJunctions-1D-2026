#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/reproduced_output"
rm -rf "$OUT"
mkdir -p "$OUT"

python "$ROOT/code/cooperative_adaptive_junction_simulator.py" \
  --outdir "$OUT/publication" \
  --precomputed-dir "$ROOT/data"

python "$ROOT/code/BB_schottky_loadline_solver.py" \
  --outdir "$OUT/BB_loadline"

# The profile validator writes beside the script by design. Copy code to an
# isolated work directory so the package source tree remains unchanged.
mkdir -p "$OUT/BB_profile_work"
cp "$ROOT/code/cooperative_adaptive_junction_simulator.py" \
   "$ROOT/code/BB_schottky_loadline_solver.py" \
   "$ROOT/code/BB_profile_validation.py" "$OUT/BB_profile_work/"
(
  cd "$OUT/BB_profile_work"
  python BB_profile_validation.py
)

echo "Reproduction complete: $OUT"
