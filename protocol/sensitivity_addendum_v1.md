# Sensitivity Addendum v1 (frozen 2026-08-04, before running)

This addendum preregisters two pre-submission sensitivity analyses. It is frozen
in git **before** any new computation runs, to guard against result-dependent
analysis selection. All predefined results will be reported regardless of sign
or magnitude. No new model architecture, no change to the main experimental
hyperparameters, no use of the test set for checkpoint selection, and no change
to the frozen split/seed lists (or their hashes) is permitted under this
addendum. RDKit, PyTorch, and data versions are those fixed for the main
analysis.

Scope control: the main experiment (5 splits x 3 seeds, fixed-final rule,
70/10/20) is NOT rerun. Only the two analyses below are added.

---

## Task A: checkpoint-selection sensitivity at the full 5-instance scale

### A.1 Motivation and scope

The main experiment uses 5 split instances x 3 seeds under the frozen
fixed-final checkpoint rule. The existing post hoc checkpoint sensitivity
(splits 0-2, validation-selected) is extended to all 5 split instances so that
checkpoint-rule sensitivity is reported at the same instance scale as the main
design. The training recipe, data, seeds, and partitions are unchanged; only
splits 3 and 4 are newly trained.

### A.2 Design (per protocol: grouped-random, scaffold-grouped)

- Instances: 0-4 (frozen global_split3, seed = SEED_BASE + inst)
- Seeds: 3 per instance (frozen)
- Partitions: the 10% calibration partition is split deterministically
  (split_cal, seed + 999) into 5% validation + 5% temperature-calibration
- Models: one MTL (8 heads) + eight architecture-matched STL per (instance, seed)
- Training: EPOCHS = 60, validation NLL evaluated every 10 epochs; model state
  saved at every evaluated epoch (checkpoints at epochs 10, 20, ..., 60)
- Temperatures: fitted per model on its own 5% temperature-calibration subset,
  applied to its own test partition (strict per-model rule, unchanged)

### A.3 Checkpoint rules (all evaluated on the same saved checkpoints)

1. **fixed-final**: the final epoch (60) for all models.
2. **global-vs-global**: MTL selects one common epoch minimizing the mean over
   the eight heads of their own-endpoint validation NLL; the eight STL models
   select one common epoch minimizing the mean over endpoints of their
   endpoint-specific validation NLL.
3. **endpoint-vs-endpoint**: each MTL head selects the epoch minimizing its own
   endpoint's validation NLL; the corresponding STL model selects the epoch
   minimizing the same endpoint's validation NLL.
4. **original hybrid** (retained as an additional sensitivity, not primary):
   MTL selects one common epoch (head-mean validation NLL); each STL selects
   its own endpoint-specific epoch.

Rules 2 and 3 are the symmetric rules reported as primary in the addendum;
rule 4 matches the previously reported post hoc analysis.

### A.4 Estimands and aggregation

- Molecule-level log-loss contrast Delta = NLL_STL - NLL_MTL (per rule, per
  protocol), endpoint-macro means; calibrated variants with per-model
  temperatures as in A.2; protocol contrast Gamma = Delta_scaffold - Delta_random.
- Aggregation: seeds are averaged **within** each split; the statistical unit is
  the split instance (5 units). We report the five split-level Gamma values,
  the number of positive splits, and a split-instance block bootstrap 95% CI.
- No per-(instance, seed) unit is treated as independent (seeds share the test
  set within a split).

### A.5 Predefined outputs

- results/checkpoint_5split_summary.csv: per rule x protocol: raw/calibrated
  Delta_random, Delta_scaffold, Gamma, positive-split count, bootstrap CI,
  selected-epoch distribution.
- results/checkpoint_selected_epochs.csv: per (instance, seed, rule, model):
  selected epoch.
- figures/fig_checkpoint_learning_curves.pdf: validation NLL vs epoch (MTL and
  STL means) for random and scaffold protocols (panels A, B).
- figures/fig_checkpoint_rule_comparison.pdf: five split-level Gamma per rule
  with means (panel C).
- SI LaTeX table with the summary of A.4 for the four rules.

### A.6 Reporting commitment

- If the mean Gamma is positive under all rules with a majority of splits
  positive: state "mean protocol interaction remained positive across
  fixed-final and both symmetric validation-selection rules at matched
  five-instance scale".
- If split-level signs are heterogeneous: state that explicitly; do not claim
  robustness.
- If mean Gamma is near zero or negative under any rule: report it and revise
  the main-text claim accordingly ("both the absolute contrast and its protocol
  interaction are checkpoint-rule dependent"); do not filter or re-select rules.

---

## Task B: chemical-entity identity collision audit

### B.1 Motivation and scope

The primary allocation keys molecule identity on exact RDKit canonical isomeric
SMILES. Because the pipeline applies no salt removal, tautomer normalization,
or charge normalization (SI S9), exact-string identity is not the same as
standardized-chemical-entity identity. This audit quantifies how much
cross-partition exposure exists under stricter chemical-entity definitions and
whether the main conclusions survive excluding affected test observations. No
retraining is performed unless the audit requires it (B.5).

### B.2 Identity levels

- **Level 0 (primary key)**: RDKit canonical isomeric SMILES (as used).
- **Level 1 (standardized parent key)**: deterministic pipeline on the fixed
  RDKit version: Cleanup -> FragmentParent (largest parent fragment) ->
  Uncharge -> canonical tautomer (TautomerEnumerator) -> canonical isomeric
  SMILES.
- **Level 2 (InChIKey connectivity block)**: first block (14 chars) of the
  standardized parent's InChIKey. May merge stereoisomers; used ONLY as an
  upper-bound related-entity audit, never as the identity definition.

### B.3 Audit (per protocol x split instance, over the global unique-molecule
pool across all eight endpoints)

For Levels 1 and 2:
- number of distinct keys;
- number of collision groups spanning train/calibration/test (a group with
  members in more than one partition);
- test observations whose key also appears in train (count and fraction);
- test observations whose key also appears in calibration (count and fraction);
- per-endpoint breakdown of affected molecule-endpoint observations.

### B.4 No-retrain filtering sensitivity

Using the existing paired prediction tables (b3_v2_out parquets), remove test
observations whose Level-1 key appears in the train or calibration partition of
the same protocol instance, then recompute (from the remaining paired
predictions):
- Delta_random, Delta_scaffold, Gamma (raw and calibrated);
- split-level Gamma values and split-instance block bootstrap CI.

Level 2 is reported as an upper-bound audit table only (it may merge
stereoisomers; it is not used for the filtering sensitivity).

### B.5 Retraining criterion

Retraining with a standardized-key global allocation would be undertaken only
if: collided test observations exceed a material fraction, or the filtered
Gamma changes sign or its CI excludes the unfiltered estimate. Otherwise the
audit is reported as a table in the SI with the filtering sensitivity.

### B.6 Predefined outputs

- results/chemical_identity_collision_audit.csv: per protocol x instance x
  level: keys, collision groups, train/test and cal/test overlaps (counts and
  fractions), per-endpoint affected counts.
- results/chemical_identity_filtered_contrasts.csv: raw/calibrated
  Delta/Gamma before and after Level-1 filtering, split-level Gammas, CIs.
- SI LaTeX table.

---

## Unit tests (added with this addendum)

- Exact-key isolation: for every (protocol, instance), no test molecule's exact
  canonical SMILES appears in any endpoint's training partition (assertion on
  the frozen split manifests).
- Level-1 key determinism: standardization is reproducible for a fixed input
  SMILES set (same outputs across two runs).

## Frozen environment

- RDKit 2024.09.6, PyTorch (installed version), Python 3.10, data files as
  archived in the repo; split/seed hashes unchanged from the main analysis.
