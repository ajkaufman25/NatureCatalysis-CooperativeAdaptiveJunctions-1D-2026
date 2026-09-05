# Integration-only changes from the August 30 source of truth

The file under `provenance/` is an exact copy of the August 30 model-code source of truth before final integration.

The final package makes no physics changes. The following packaging changes were made:

1. **Continuous figure numbering.** The experimental SI already contains Figure S14. Model Figures S14-S18 were therefore renumbered S15-S19 in the final SI, model source, simulator output filenames, and figure proof.
2. **Precomputed data filename compatibility.** The simulator now accepts both the internal legacy continuation filenames and the publication CSV filenames `AB_catalyst_off_Voc_sweep.csv` and `BA_catalyst_off_Voc_sweep.csv`. This changes only file discovery, not computation.
3. **Reproducibility outputs.** Added independent BB validation outputs, a validation report, README, and SHA-256 manifest.

The complete SI, manuscript text, experimental figures, and document preflight outputs are intentionally excluded from this repository.
