# Deviations Log

| Date (JST) | Deviation | Reason | Status |
|-----------|-----------|--------|--------|
| 2026-08-03 | v1 two-way 80/20 allocation + cross-instance calibration replaced by 70/10/20 three-way + strict per-model calibration | Reviewer objection: temperatures were shared across independently trained models and calibration/test could overlap at the molecule level | Replaced before confirmatory rerun |
| 2026-08-03 | Scaffold protocol upgraded from 3 to 5 split instances | Instance-count asymmetry with the random protocol; 3-instance novelty reversal did not survive 5 instances | Replaced before confirmatory rerun |
| 2026-08-03 | Downsampling intervention re-run under the three-way design | v1 numbers came from the old 80/20 design and could not be mixed with v2 main results | Re-run (b3_ds_v2.py) |

The contaminated development pass (per-endpoint splits + loss-scaling bug) is
excluded from all confirmatory statistics and reported in Supporting Information S2.


## 2026-08-04: checkpoint sensitivity training defect (sensitivity addendum)

The earlier post hoc checkpoint-sensitivity script (b3_checkpoint_sens.py,
three-instance scale) trained every MTL head on a single endpoint's task
(`train_with_ckpt(tasks[0][0], tasks[0][1], ...)`), producing the negative
absolute contrasts previously reported as "validation selection reverses the
absolute contrast". The corrected five-instance analysis
(b3_checkpoint_5split.py, frozen protocol/sensitivity_addendum_v1.md) trains
each MTL head on its own endpoint's task and supersedes those numbers: the
absolute contrast is attenuated but remains positive under all checkpoint
rules, and the protocol interaction is positive under every rule.

## 2026-08-07: identity-audit filtering aggregation correction (sensitivity addendum)

The no-retrain filtering sensitivity in b3_identity_audit.py first averaged
the stored cell-constant `d`/`d_cal` columns of the prediction tables, which
made the Level-1 filtering appear inert (contrast unchanged at `+0.083`
before and after filtering). The corrected aggregation (2026-08-07) recomputes
per-observation contrasts from the frozen `p_stl`/`p_mtl`/`p_stl_cal`/
`p_mtl_cal` columns and averages equal-weight (instance, seed) cells,
matching the main analysis. Corrected result: split-level Gamma moves from
`+0.083` unfiltered to `+0.071` after filtering (per-split `+0.098, +0.080,
+0.084, -0.010, +0.102`; filtered 95% split-instance bootstrap interval
`[+0.029, +0.097]`, seed 0), so the preregistered no-retraining criterion in
protocol/sensitivity_addendum_v1.md (retrain only if the filtered contrast
changes sign or its interval excludes the unfiltered estimate) is not
triggered. The corrected numbers replace those in b3_idaudit_out/
chemical_identity_filtered_contrasts.csv and are reported in Supporting
Information S12. No split, seed, training recipe, or prediction table changed.
