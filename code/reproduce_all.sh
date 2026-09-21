#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/reproduced_output/full_suite}"
# Modest BLAS thread counts avoid oversubscription for the collocation solves.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLBACKEND=Agg
python "$ROOT/code/reproduce.py" --full-sweeps --diagnostics --outdir "$OUT/publication"
python "$ROOT/code/BB_schottky_loadline_solver.py" --outdir "$OUT/BB_loadline"
python "$ROOT/code/BB_profile_validation.py" --outdir "$OUT/BB_profiles"
python "$ROOT/code/validate_convergence.py" --outdir "$OUT/convergence"
python -m unittest discover -s "$ROOT/tests" -v
# The S15 schematic is already supplied. Rebuild separately when TeX is available:
# python "$ROOT/code/build_framework.py" --outdir "$OUT/publication/figures"
echo "Reproduction and validation completed: $OUT"
