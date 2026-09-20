# One-Dimensional Cooperative Adaptive Junction Model

Reproducible model code, generated figures, validation data, and provenance for:

> **Cooperative Adaptive Junctions Govern Overall Photoelectrochemical Water Splitting**  
> Aaron Kaufman, Kaden Wheeler, Martha Kubakh, Ethan J. Crumlin, and Shannon W. Boettcher  
> *Nature Catalysis* (2026)

This repository contains the spatially resolved one-dimensional model used to compare adaptive semiconductor-catalyst contacts with conventional buried Schottky contacts. It accompanies the related [zero-dimensional interface-model repository](https://github.com/ajkaufman25/NatureCatalysis-CooperativeAdaptiveJunctions-2026).

## Scientific scope

The model treats an *n*-SrTiO₃ absorber bounded by Pt/HER and CoOₓ/OER contacts. It solves semiconductor electrostatics, electron and hole drift-diffusion, uniform photogeneration, Shockley-Read-Hall recombination, reversible semiconductor-catalyst charge transfer, and HER/OER kinetics self-consistently.

Two limiting contact boundary conditions are compared:

- **Adaptive contact (A):** the SrTiO₃ surface band energy remains pinned while the catalyst electron electrochemical potential shifts.
- **Buried contact (B):** the physical Schottky barrier remains fixed, so catalyst polarization shifts the adjacent SrTiO₃ bands one-for-one.

The four simulated architectures are AA, AB, BA, and BB.

## Repository contents

| Path | Contents |
|---|---|
| `code/` | Publication simulator, BB-control solvers, AA thickness driver, plotting script, and reproduction scripts |
| `data/` | Validated CSV inputs and publication outputs |
| `figures/` | Model Figures S15-S19 (PDF/PNG) and thickness Figures S20-S22 (supplied PDFs) |
| `data/thickness/` | Original, refined, combined, and representative-point thickness CSVs |
| `validation/thickness_reproduction/` | Supplied 62-point numerical reproduction and reference comparisons |
| `validation/` | Independent numerical checks, regenerated outputs, and spatial profiles |
| `provenance/` | August 30 model-code source of truth and the thickness-package import record |
| `VALIDATION_REPORT.md` | Numerical and figure-regression validation summary |
| `MANIFEST_SHA256.csv` | File sizes and SHA-256 checksums for release integrity |

## Requirements

- Python 3.12 or newer
- NumPy 1.24 or newer
- pandas 2.0 or newer
- SciPy 1.10 or newer
- Matplotlib 3.7 or newer

No non-standard hardware is required.

## Quick start: Windows PowerShell

Run these commands from PowerShell after cloning the repository:

```powershell
git clone https://github.com/ajkaufman25/NatureCatalysis-CooperativeAdaptiveJunctions-1D-2026.git
cd NatureCatalysis-CooperativeAdaptiveJunctions-1D-2026

py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r code\requirements.txt

powershell -ExecutionPolicy Bypass -File code\reproduce_all.ps1
```

If Python 3.12 is already the default Python installation, `python -m venv .venv` can be used in place of `py -3.12 -m venv .venv`.

## Quick start: macOS or Linux

```bash
git clone https://github.com/ajkaufman25/NatureCatalysis-CooperativeAdaptiveJunctions-1D-2026.git
cd NatureCatalysis-CooperativeAdaptiveJunctions-1D-2026

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt

bash code/reproduce_all.sh
```

## What reproduction generates

The reproduction scripts:

1. regenerate the publication model outputs in `reproduced_output/publication/`;
2. rerun the independent BB load-line calculation;
3. rerun the BB spatial-profile validation;
4. recompute the 31 global and 31 refined AA thickness points in `reproduced_output/thickness/`; and
5. regenerate Figures S20-S22 from those recomputed points in `reproduced_output/thickness_figures/`, alongside the baseline model and validation figures.

The baseline reproduction command uses precomputed AB/BA sweeps and supplied parameter-sweep tables in `data/`. The core simulator's `--full-sweeps` flag currently does not override precomputed inputs. The thickness driver separately recomputes its states without loading saved solution profiles. Both reproduction scripts replace the existing `reproduced_output/` directory when run.

## Validated architecture results

Results at 10 mW cm⁻² and 365 nm:

| Architecture | Validated result |
|---|---|
| AA | OWS: 0.945386 mA cm⁻²; catalyst separation: 1.885586 V |
| BA | OWS: 0.030379 mA cm⁻²; catalyst-off ceiling: 2.190227 V; separation: 1.676636 V |
| AB | Catalyst-off ceiling: 0.707911 V, below the 1.229 V OWS requirement |
| BB | Photovoltaic *V*<sub>OC</sub>: 0.150000 V; H₂/O₂ reverse load: −0.0499973 mA cm⁻² at 0.152745 V |

See `VALIDATION_REPORT.md` for the numerical comparisons and validation criteria.

## Model figure outputs

Figures S15-S19 are stored in `figures/` as PDF and PNG files. Figures S20-S22 are the supplied thickness PDFs; see `FIGURE_INDEX.csv` for exact paths.

## SI V10 thickness study

The supplied thickness data cover AA at 31 logarithmically spaced thicknesses from 0.01 to 10 µm and 31 refined points from 0.05 to 0.5 µm. The combined plotting/reference table is `data/thickness/AA_thickness_sweep_combined_dense.csv`; Table S9 values are in `data/thickness/AA_thickness_representative_points.csv`. The remaining supplied CSVs are retained with their original filenames for traceability.

At 1 µm, the stored current is 0.94539 mA cm⁻². The largest sampled current is 1.38513 mA cm⁻² at 3.981 µm, with an operating catalyst separation of 1.90881 V. This is a sampled AA maximum under the stated assumptions, not a universal optimum or an AA/BA comparison across thicknesses. Varying thickness changes both absorbed photon flux and transport distance.

To redraw Figures S20-S22 from the supplied data **without running the simulation**, run:

```bash
python code/AA_thickness_crossover_report.py --outdir reproduced_output/thickness_figures
```

To recompute thickness states in a future run, use:

```bash
python code/AA_thickness_sweep.py --outdir reproduced_output/thickness
python code/AA_thickness_crossover_report.py --data-file reproduced_output/thickness/AA_thickness_sweep_reproduced.csv --outdir reproduced_output/thickness_figures
```

The SI V10 repository update imported the supplied results and figure PDFs without rerunning simulations. The numerical validation in `validation/thickness_reproduction/` was supplied in the September 9 package. See `VALIDATION_REPORT.md` for its results and the scope of the import checks.

## Citation

If you use this software, please cite:

> A. Kaufman, K. Wheeler, M. Kubakh, E. J. Crumlin, and S. W. Boettcher, “Cooperative Adaptive Junctions Govern Overall Photoelectrochemical Water Splitting,” *Nature Catalysis* (2026).

## License

This repository is released under the MIT License. See `LICENSE`.
