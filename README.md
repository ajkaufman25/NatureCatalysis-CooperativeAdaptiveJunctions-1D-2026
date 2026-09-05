# One-Dimensional Cooperative Adaptive Junction Model

Reproducible code, data, validation outputs, and Supporting Information for:

> **Cooperative Adaptive Junctions Govern Overall Photoelectrochemical Water Splitting**  
> Aaron Kaufman, Kaden Wheeler, Ethan J. Crumlin, and Shannon W. Boettcher  
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
| `code/` | Publication simulator, independent BB-control solvers, dependency list, and reproduction scripts |
| `data/` | Validated CSV inputs and publication outputs |
| `validation/` | Independent numerical checks, regenerated outputs, spatial profiles, and PDF preflight results |
| `SI/` | Complete LaTeX sources, compiled SI PDFs, model proof, figure proof, and figure assets |
| `provenance/` | Exact August 30 publication-facing source files retained before packaging-only changes |
| `VALIDATION_REPORT.md` | Numerical and figure-regression validation summary |
| `MANIFEST_SHA256.csv` | File sizes and SHA-256 checksums for release integrity |

## Requirements

- Python 3.12 or newer
- NumPy 1.24 or newer
- pandas 2.0 or newer
- SciPy 1.10 or newer
- Matplotlib 3.7 or newer
- For rebuilding PDFs: `latexmk` plus a complete TeX distribution containing the `soul`, `bookmark`, `subcaption`, `microtype`, and `footnotehyper` packages (for example, MiKTeX on Windows or TeX Live with `texlive-latex-extra` on Ubuntu/Debian)

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
3. rerun the BB spatial-profile validation; and
4. compile the full SI, standalone model section, and one-figure-per-page proof.

The validated continuation CSVs in `data/` are used for the numerically stiff AB and BA sweeps.

## Validated architecture results

Results at 10 mW cm⁻² and 365 nm:

| Architecture | Validated result |
|---|---|
| AA | OWS: 0.945386 mA cm⁻²; catalyst separation: 1.885586 V |
| BA | OWS: 0.030379 mA cm⁻²; catalyst-off ceiling: 2.190227 V; separation: 1.676636 V |
| AB | Catalyst-off ceiling: 0.707911 V, below the 1.229 V OWS requirement |
| BB | Photovoltaic *V*<sub>OC</sub>: 0.150000 V; H₂/O₂ reverse load: −0.0499973 mA cm⁻² at 0.152745 V |

See `VALIDATION_REPORT.md` for the numerical comparisons and validation criteria.

## Supporting Information outputs

- `SI/Cooperative_Adaptive_Junctions_Supporting_Information_FINAL.pdf` — compiled full SI
- `SI/Cooperative_Adaptive_Junctions_model_section_FINAL.pdf` — standalone model-section proof
- `SI/Cooperative_Adaptive_Junctions_Figure_Proof_FINAL.pdf` — one-figure-per-page proof of Figures S1–S19

The final integration renumbered the model figures from S14–S18 to S15–S19 because the experimental SI already contained Figure S14. No governing equations, physical parameters, kinetic laws, or validated operating points were changed. See `CHANGES_FROM_AUG30_SOURCE.md` for the full packaging record.

## Citation

If you use this software, please cite:

> A. Kaufman, K. Wheeler, E. J. Crumlin, and S. W. Boettcher, “Cooperative Adaptive Junctions Govern Overall Photoelectrochemical Water Splitting,” *Nature Catalysis* (2026).

## License

This repository is released under the MIT License. See `LICENSE`.
