# One-Dimensional Cooperative Adaptive Junction Model

Model code, figures, numerical data, and validation for:

> **Cooperative Adaptive Junctions Govern Overall Photoelectrochemical Water Splitting**  
> Aaron Kaufman, Kaden Wheeler, Martha Kubakh, Ethan J. Crumlin, and Shannon W. Boettcher

This repository contains the one-dimensional semiconductor model used in the Supplementary Information (SI). It includes the code and data for the operating-state calculations, parameter sweeps, thickness study, and model Figures S15–S19.

## Scientific scope

The model treats an *n*-SrTiO₃ absorber between Pt/HER and CoOₓ/OER contacts. It solves Poisson's equation, electron and hole drift–diffusion, and Shockley–Read–Hall recombination with uniform photogeneration, reversible semiconductor–catalyst charge transfer, and HER/OER kinetics. The catalyst potentials are determined by steady-state current balance.

Two contact conditions are compared:

- **Adaptive (A):** the semiconductor surface band energy is pinned while the catalyst potential changes.
- **Buried (B):** the Schottky barrier is fixed, so changes in catalyst potential shift the semiconductor surface bands.

The first letter identifies the HER contact and the second identifies the OER contact. The four architectures are AA, AB, BA, and BB. Finite-current forward water splitting is calculated for AA and BA; dark equilibria are calculated for all four architectures. BB open circuit and reverse HOR/ORR loading are evaluated separately.

## Requirements

The supplied validation results were generated with **Python 3.13.5**. Use Python 3.13 and `code/requirements-tested.txt` to install the recorded package versions:

| Package | Recorded version |
|---|---:|
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |
| PyMuPDF | 1.26.7 |

`code/requirements.txt` lists minimum dependency versions for other environments. The commands below use the recorded versions.

Git is needed to clone the repository. You can also download and extract the repository ZIP. Rebuilding Figure S15 additionally requires `pdflatex` with the `standalone`, TikZ/PGF, and Latin Modern packages. The numerical studies and Figures S16–S19 do not require LaTeX.

## Quick start: Windows PowerShell

Install Python 3.13 and Git, then open PowerShell. If you downloaded a ZIP, open PowerShell in the extracted repository folder and skip the first two commands.

```powershell
git clone https://github.com/ajkaufman25/NatureCatalysis-CooperativeAdaptiveJunctions-1D-2026.git
cd NatureCatalysis-CooperativeAdaptiveJunctions-1D-2026

py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r code\requirements-tested.txt

powershell -ExecutionPolicy Bypass -File code\reproduce_all.ps1
```

The final command runs the full numerical suite and writes results to `reproduced_output/full_suite/`. It includes fresh model solves and can take longer than redrawing figures from the supplied data.

For a later session, return to the repository folder and reactivate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Quick start: macOS or Linux

Install Python 3.13 and Git, then run the following commands in a terminal. If you downloaded a ZIP, enter the extracted repository folder and skip the first two commands.

```bash
git clone https://github.com/ajkaufman25/NatureCatalysis-CooperativeAdaptiveJunctions-1D-2026.git
cd NatureCatalysis-CooperativeAdaptiveJunctions-1D-2026

python3.13 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r code/requirements-tested.txt

bash code/reproduce_all.sh
```

For a later session, return to the repository folder and run `source .venv/bin/activate`.

## What the full numerical suite runs

Both `reproduce_all` scripts perform the same sequence:

| Study or check | Scope |
|---|---|
| AA and BA operating states | Solve catalyst current balance and semiconductor profiles at finite forward OWS current |
| Dark equilibria | Solve AA, AB, BA, and BB at a common 0 V catalyst potential, with generation off and Faradaic exchange disabled |
| Native band bending | Five mean band-bending values (0.70, 0.85, 0.925, 1.00, 1.15 V) and six asymmetries (0, 0.05, 0.10, 0.15, 0.20, 0.30 V); 25 numerical solutions and five zero-current symmetry assignments |
| Interfacial transfer | Equal electron/hole transfer velocities of 0.1, 1, 10, 100, and 1000 m s⁻¹ |
| Illumination intensity | 1, 3, 5, 10, 20, and 30 mW cm⁻² at 365 nm |
| AA thickness | 31 logarithmically spaced points from 0.01 to 10 µm and 31 refined points from 0.05 to 0.5 µm |
| BB diagnostics | Semiconductor current–voltage curve, open circuit, and reverse HOR/ORR load intersection |
| Independent BB load line | Separate load-line implementation and endpoint checks |
| BB profiles | Re-solve at absolute catalyst potentials and check current balance and gauge consistency |
| AA/BA convergence | Tighter collocation settings at fixed operating catalyst potentials, plus an SRH integration check |
| Regression tests | Equations, parameter defaults, units, boundary signs, and fresh AA/BA endpoint checks |

The publication run also produces Figures S16–S19 and comparisons with the reference data. Figure S19 uses the 31 global thickness points; the additional 31 refined points remain in the numerical output.

Figure S15 is a schematic. Its supplied PDF and PNG are in `figures/`; rebuild it separately using the command below if needed.

The selected AA, AB, and BA catalyst-off separations in SI Table S8 are supplied comparison values. The SI does not specify the additional selection constraints needed to reconstruct those states uniquely. They are copied into `selected_catalyst_off_states.csv` with source labels. Historical AB/BA catalyst-off sweeps remain in `data/reference/` and are not recomputed by the full suite.

## Output folders

With the default full-suite command, results are written under `reproduced_output/full_suite/`:

| Folder | Contents |
|---|---|
| `publication/data/` | Operating and dark states, spatial profiles, sensitivity sweeps, and thickness tables |
| `publication/figures/` | Figures S16–S19 as PDF and PNG |
| `publication/validation/` | Baseline and sensitivity endpoint checks and reference comparisons |
| `publication/thickness/` | All 62 thickness points, endpoint checks, comparisons, and summary |
| `publication/diagnostics/` | BB reverse-load results, profiles, and semiconductor current–voltage curve |
| `BB_loadline/` | Independent BB load-line results and checks |
| `BB_profiles/` | BB open-circuit and loaded profiles, plots, and validation results |
| `convergence/` | AA/BA refinement comparisons and endpoint checks |

`publication/run_metadata.json` records the environment, model inputs, calculation groups, and completion status. Unit-test results appear in the terminal. A failed numerical check stops the full-suite script.

### Repeating a run

Choose a new output folder for each full-suite run. The publication driver refuses to write into a nonempty folder by default.

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File code\reproduce_all.ps1 -OutDir reproduced_output/full_suite_run2
```

macOS or Linux:

```bash
bash code/reproduce_all.sh reproduced_output/full_suite_run2
```

When using `reproduce.py` directly, `--overwrite` allows writing into an existing output folder but does not remove old files. Use a new folder when comparing runs.

## Run individual studies

Run the commands below from the repository root with the virtual environment active. These Python commands work in PowerShell, macOS, and Linux.

### Baseline operating and dark states

```bash
python code/reproduce.py --outdir reproduced_output/baseline
```

This solves AA/BA operating states and all four dark equilibria, checks the baseline profiles, and generates Figures S16–S19. It uses stored sensitivity and thickness tables for the corresponding figures. It still runs the baseline simulation.

### All publication sweeps and BB diagnostics

```bash
python code/reproduce.py --full-sweeps --diagnostics --outdir reproduced_output/publication
```

This recomputes the baseline, native band-bending, transfer-velocity, intensity, and thickness studies, plus the BB reverse-load diagnostic. Reference tables are used for comparison. Add `--no-plots` to skip figure rendering. The separate BB validators, convergence check, and unit tests are run by the full-suite scripts or the commands below.

### AA thickness study

```bash
python code/AA_thickness_sweep.py --outdir reproduced_output/thickness
```

The default run solves all 62 thickness points. To calculate selected thicknesses in µm:

```bash
python code/AA_thickness_sweep.py --lengths-um 0.1 1 4 10 --outdir reproduced_output/thickness_selected
```

All requested endpoints undergo numerical checks. Reference comparisons are made only where a matching reference thickness exists.

To draw Figure S19 from the complete thickness run:

```bash
python code/AA_thickness_crossover_report.py --data-file reproduced_output/thickness/AA_thickness_sweep.csv --outdir reproduced_output/thickness/figures
```

### Independent BB load-line study

```bash
python code/BB_schottky_loadline_solver.py --outdir reproduced_output/BB_loadline
```

### BB spatial-profile validation

```bash
python code/BB_profile_validation.py --outdir reproduced_output/BB_profiles
```

This writes open-circuit and reverse-load profiles, endpoint checks, and diagnostic PDF plots. Add `--no-plots` to skip the plots.

### AA/BA convergence check

```bash
python code/validate_convergence.py --outdir reproduced_output/convergence
```

This first calculates fresh AA/BA operating states, then refines their spatial solutions at fixed catalyst potentials and checks the SRH integration. It does not repeat the outer catalyst-current root at the tighter settings.

### Regression tests

```bash
python -m unittest discover -s tests -v
```

The tests include fresh AA/BA numerical solves as well as equation and notation checks.

## Redraw figures without running the simulation

To redraw Figures S16–S19 from the supplied publication CSVs:

```bash
python code/plot_publication.py --data-dir data/publication --outdir redrawn_figures
```

To redraw them from a full-suite run:

```bash
python code/plot_publication.py --data-dir reproduced_output/full_suite/publication/data --outdir reproduced_output/full_suite/redrawn_figures
```

To redraw only the thickness figure from supplied data:

```bash
python code/AA_thickness_crossover_report.py --outdir redrawn_figures/thickness
```

These commands read existing CSVs and do not solve the semiconductor model.

To rebuild the Figure S15 schematic with LaTeX:

```bash
python code/build_framework.py --outdir redrawn_figures
```

The schematic build writes PDF and PNG files. Without LaTeX, use the supplied Figure S15 files in `figures/`.

| Figure | Contents |
|---|---|
| S15 | Model framework and boundary conditions |
| S16 | AA and BA operating energy profiles and current balance |
| S17 | Sensitivity to native facet band bending |
| S18 | Sensitivity to transfer velocity and illumination intensity |
| S19 | AA thickness dependence of current balance and catalyst-potential separation |

See [FIGURE_INDEX.csv](FIGURE_INDEX.csv) for filenames, inputs, and plotting scripts.

## Baseline results and units

The supplied publication data give the following operating states at 1 µm thickness, 10 mW cm⁻², and 365 nm:

| Architecture | OWS current (mA cm⁻²) | Catalyst-potential separation (V) |
|---|---:|---:|
| AA | 0.945386 | 1.885586 |
| BA | 0.030379 | 1.676636 |

The separate BB diagnostic gives approximately 0.150000 V open circuit and a reverse current of −0.0499973 mA cm⁻² at a catalyst separation of 0.152745 V. This reverse current corresponds to HOR/ORR.

Publication CSVs report electronic energies in **eV**, catalyst potentials in **V vs RHE**, current densities in **mA cm⁻²**, and spatial coordinates in **µm**. The numerical electron-energy mapping is `E_eV = -V_V` relative to the RHE electron level. The catalyst voltage separation is `V_cat,OER - V_cat,HER`; the corresponding energy separation uses `E_cat,HER - E_cat,OER`.

Definitions and sign conventions are in [docs/NOTATION.md](docs/NOTATION.md). Column definitions are in [docs/OUTPUT_SCHEMA.csv](docs/OUTPUT_SCHEMA.csv).

## Repository contents

| Path | Contents |
|---|---|
| `code/` | Simulator, study drivers, plotting tools, validators, and dependency lists |
| `data/publication/` | Supplied data with SI energy/voltage notation |
| `data/reference/` | Numerical reference tables and profiles used for comparisons |
| `figures/` | Model Figures S15–S19 in PDF and PNG |
| `docs/` | Notation, numerical methods, output schema, and SI LaTeX source |
| `tests/` | Equation, notation, and numerical regression tests |
| `validation/si_alignment/` | Supplied study outputs, validation records, and run logs |
| `provenance/` | Original source code, reference figures, and historical validation material |
| `MANIFEST_SHA256.csv` | Package file sizes and checksums |

Read [docs/NUMERICAL_METHODS.md](docs/NUMERICAL_METHODS.md) for the equations, nondimensionalization, boundary conditions, solver settings, continuation methods, and acceptance criteria.

The one-dimensional model accompanies the related [zero-dimensional interface-model repository](https://github.com/ajkaufman25/NatureCatalysis-CooperativeAdaptiveJunctions-2026).

## Citation and license

Citation metadata and author information are provided in [CITATION.cff](CITATION.cff). If you use this model, cite the associated work:

> A. Kaufman, K. Wheeler, M. Kubakh, E. J. Crumlin, and S. W. Boettcher, “Cooperative Adaptive Junctions Govern Overall Photoelectrochemical Water Splitting.”

The code is distributed under the [MIT License](LICENSE).
