# Model Validation Report

Validation date: 2026-08-31

## 1. Publication solver reproduction

The packaged simulator was run as:

    python code/cooperative_adaptive_junction_simulator.py \
      --outdir validation/publication_reproduction \
      --precomputed-dir data

Recomputed baseline:

- absorbed photon-equivalent generation: 1.860915 mA cm^-2
- AA OWS current: 0.945386 mA cm^-2
- AA catalyst separation: 1.885586 V
- BA catalyst-off ceiling: 2.190227 V
- BA OWS current: 0.030379 mA cm^-2
- AB catalyst-off ceiling: 0.707911 V
- BB photovoltaic Voc: 0.150000 V
- BB reverse-load current: -0.0499973 mA cm^-2
- BB loaded terminal separation: 0.152745 V

All 13 publication CSV files regenerate with identical dimensions/columns and maximum finite numerical differences below 1e-10; most are byte-for-value identical. The largest observed roundoff differences are approximately 8e-25.

## 2. Figure regression

The five model figures were regenerated from the final simulator.

## 3. Independent BB load-line validation

The independent `BB_schottky_loadline_solver.py` calculation gives:

- Phi_H = 0.955212010850 eV
- Phi_O = 1.105212010850 eV
- DeltaPhi = 0.150000000000 eV
- Voc = 0.149999999982 V
- reverse current magnitude = 49.997341312 microA cm^-2
- U_H = 0.597324955487 V vs RHE
- U_O = 0.750070018175 V vs RHE
- U_O-U_H = 0.152745062688 V
- scalar current-balance residual = approximately 4e-12 A m^-2

## 4. Independent BB spatial/gauge validation

`BB_profile_validation.py` re-solves the full Poisson + electron/hole drift-diffusion + SRH boundary-value problem at the absolute loaded catalyst potentials. The resulting checks include:

- HER current-balance residual: ~6e-12 A m^-2
- OER current-balance residual: ~6e-12 A m^-2
- total-current spatial span: ~7e-15 A m^-2
- BVP maximum RMS residual: ~1e-8
- maximum gauge-invariance U_CB shift error: ~4e-11 V
- Voc recheck current: ~4e-9 microA cm^-2

The complete numerical summary and spatial profiles are under `validation/BB_profile_validation/`.

## SI V10 thickness package: supplied historical validation

The September 9, 2026 package supplies `validation/thickness_reproduction/AA_thickness_sweep_reproduced.csv`, `reference_comparison.csv`, and `reproduction_summary.json`. Its report records independent reproduction of all 62 endpoints using the unchanged publication simulator. These calculations were **not rerun during the SI V10 repository update**.

The supplied summary reports:

- Maximum OWS-current difference: 2.08167e-15 mA cm^-2.
- Maximum operating-separation difference: 1.44635e-11 V.
- Maximum BVP residual: 2.99991e-7.
- Maximum integrated current-budget residual: 5.35088e-7 A m^-2.
- Maximum catalyst current-balance residual: 2.94564e-10 A m^-2.

The historical acceptance criteria were absolute reference differences below 1e-7 in each quantity's reported units, BVP residual at most 3.1e-7, integrated current-budget residual below 1e-6 A m^-2, and catalyst current-balance residual below 1e-8 A m^-2. The supplied results meet these criteria. Numerical reproduction does not establish experimental accuracy or a universal optimum thickness.

## SI V10 import checks (no simulation execution)

The import preserves the supplied thickness CSVs, numerical validation files, and Figure S20-S22 PDFs byte-for-byte. The core publication simulator remains byte-identical to both the archive and the previous repository version (SHA-256 `d678729bef6a903d614aa6b7528213b4787c6b5daf68504bf7c075caef56e353`).

Checks cover CSV readability, 31 global plus 31 refined reference points, agreement with SI V10 Table S9 at its displayed precision, stored reference-comparison values and diagnostics, figure/index paths, Python syntax, shell syntax, and release hashes. No solver or figure-generation script was executed. PowerShell changes were inspected but not executed.

The plotting script has packaging-only changes: its default output is `reproduced_output/thickness_figures/`, and `--data-file` accepts the combined CSV written by the thickness driver. Both reproduction entry points now call the thickness driver and plot its resulting data. Imported-file source hashes and these adaptations are recorded in `provenance/SI_V10_import.json`.
