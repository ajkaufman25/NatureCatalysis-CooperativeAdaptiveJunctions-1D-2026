# Immutable numerical references

These are upstream regression inputs, retaining their original numerical precision, column names and historical diagnostic labels. They are not the publication-facing E/eV and V/V schema. Source hashes and the upstream commit are recorded in `provenance/source_verification.json`.

The canonical `thickness/AA_thickness_sweep_combined_dense.csv` contains the original 31 global and 31 refined thickness points. Redundant export variants from the earlier package are not separate physical data sets and are not required by the updated drivers.

`AB_catalyst_off_Voc_sweep.csv` and `BA_catalyst_off_Voc_sweep.csv` contain historical diagnostic families/asymptotic sweeps. Their `Voc` names do not imply a unique photovoltaic open circuit for an ideal adaptive/floating architecture. They are not the selected comparison states in current SI Table S8 and are not read as inputs to the AA/BA operating-current roots.

No reference CSV is overwritten by a reproduction command. New numerical results are placed in its chosen output directory.
