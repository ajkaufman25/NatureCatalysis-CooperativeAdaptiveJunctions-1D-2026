# Conservative SI-alignment update — 2026-09-20

Base commit: `afd7e86a5976042ae2eb17f325aa49f98544d45e` of `ajkaufman25/NatureCatalysis-CooperativeAdaptiveJunctions-1D-2026`.

## Preserved

The physical Poisson, drift–diffusion, SRH, carrier-transfer and Faradaic equations; physical parameter defaults; contact architectures; original numerical scaling; baseline AA/BA load-line equations; and global/refined parameter grids are unchanged. Exact upstream code and regression CSVs are preserved under `provenance/upstream_code/` and `data/reference/`. The SI source itself is unchanged.

## Publication-facing changes

Energy/voltage helpers and CSV exports now distinguish E/eV from V/V and preserve the required electron sign. Band-gap and barrier conversions are marked explicitly. Current-balance terminology, geometry signs, true dark-reference labels and the BA interpretation match the SI. Current model figures are S15–S19; retired figure numbering is confined to provenance. S15 comes from literal SI TikZ; S16–S19 use the supplied SI panel layouts and freshly calculated numerical data.

The SI's selected AA/AB/BA catalyst-off states remain supplied values. Historical fitted/asymptotic diagnostic separations are not substituted for them. BB reverse loading remains a separate diagnostic.

## Numerical safeguards and documentation

Added equation-by-equation nondimensionalization, units, boundary signs, continuation/root explanation, endpoint criteria and data provenance. Added a true common-Fermi dark-reference method without changing the older numerical starting-state API. The full-sweep option now actually recomputes the sweeps. Initial guesses for difficult sensitivity points are continued from freshly solved nearby parameters. Endpoint equations and target parameters are unchanged.

Failed final AA refinement and failed independent BA balance checks now raise errors. BB caches honor tighter requested tolerances/meshes. The optional mixed-contact root returns an explicitly re-solved endpoint instead of an approximately neighbouring cached state. These are failure-reporting/accuracy safeguards, not changes to the physical model.

Added automated comparison with immutable upstream equations and profiles, fresh AA/BA tests, four-sign tests, BA physical-definition tests and release integrity checks. Full CSV precision is retained. Package versions and actual run logs accompany the validation report.

## Boundaries of this update

The input SI does not specify selection constraints for its floating AA/AB/BA catalyst-off entries. None is invented. Historical deep-reverse-bias mixed-contact sweeps are retained as source data, outside current-figure regeneration. Attempts to rerun every deep-bias endpoint encountered absolute boundary-tolerance failures; those runs are not reported as passes. No tunnelling, degenerate statistics, leakage, extra selectivity, finite catalyst capacity or new physical mechanism has been added to make a diagnostic converge.

The full repository history is not included. Original duplicate thickness export variants are represented by the byte-verified canonical 62-point reference table. Original numerical source and relevant validation/figure material remain available in provenance; all old filenames requiring removal/replacement are described in UPLOAD_NOTES.md.
