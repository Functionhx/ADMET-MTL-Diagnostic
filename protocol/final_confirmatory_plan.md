# Final Confirmatory Plan (frozen 2026-08-03, before the three-way rerun)

This plan was frozen in `b3_main_v2.py` (created 2026-08-03 14:31 JST) and
recorded in `docs/PROTOCOL.md` (commit 5c2cd1e, 2026-08-03 14:47 JST) before
the confirmatory rerun of the corrected three-way design (random-protocol run
started 14:39, scaffold run 15:20).

## Frozen elements
- Global three-way allocation: 70% train / 10% calibration / 20% test
  (random: molecule-level; scaffold: Bemis-Murcko scaffold-grouped), 5 instances x 3 seeds
- Strict per-model temperature calibration (T fitted and applied within the
  same trained model; no cross-model temperature sharing)
- Estimands: molecule-level log-loss contrast, endpoint-macro Delta, protocol contrast Gamma
- Data curation: canonical-SMILES round-trip, first-occurrence dedup rule,
  13 conflicting-label molecules (hERG 3, BBB_Martins 10) with sensitivity analysis
- Analysis code: b3_main_v2.py, b3_ds_v2.py, b3_gnn_baseline.py (5x3),
  b3_controls.py, b3_r2c_reviewer_analysis.py

## Deviations from the development design (recorded in deviations.md)
## Post hoc analyses (recorded in posthoc_analyses.md)
