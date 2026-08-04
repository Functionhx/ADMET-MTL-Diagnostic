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
