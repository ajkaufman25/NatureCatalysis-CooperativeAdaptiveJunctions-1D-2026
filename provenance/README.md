# Provenance and immutable reference material

`source_verification.json` records the inspected upstream commit and SHA-256 checks of source code/data recovered from the supplied model archive and GitHub text fetches. The canonical 62-point thickness file and original thickness driver were checked byte-for-byte against the GitHub manifest. The source SI hash identifies the exact unchanged reference document for this release.

`upstream_code/` preserves the four executable numerical source modules used for equation/default regression. These are historical source, not the public E/V interface. `upstream_figures/` preserves original model figures with their old numbering. `si_reference_figures/` preserves the supplied current SI model figures and their plotting script for visual/layout comparison. Historical source terminology inside these directories is intentional.

`historical_validation/` contains validation material supplied with the original model archive. Its dates and claims apply to those original runs. Newly performed numerical solves and acceptance checks are stored in `validation/si_alignment/`; they are not inferred from historical reports.

`data/reference/` contains original numerical reference values and column names without rounding or reinterpretation. `data/publication/` contains the present SI-named outputs. The one-electron E/eV versus V/V mapping is documented separately and is tested; no legacy CSV header is silently relabelled as an energy.
