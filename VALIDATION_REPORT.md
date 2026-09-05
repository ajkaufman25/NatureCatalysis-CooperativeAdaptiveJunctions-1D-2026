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

The five model figures were regenerated from the final simulator after changing only their SI numbering from S14-S18 to S15-S19. Raster comparison at 160 dpi against the August 30 source PDFs gives 0.0% changed pixels for all five figures. Thus the model graphics are visually identical; only filenames/figure numbers changed.

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

## 5. Figure numbering

The five model figures were assigned the continuous SI labels S15-S19 during final integration. The complete SI and its document-preflight outputs are intentionally excluded from this repository.
