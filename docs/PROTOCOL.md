# Frozen Analysis Protocol (v2, 2026-08-03)

This document freezes the confirmatory analysis protocol for the manuscript
*A Leakage-Controlled Diagnostic Study of How Evaluation Protocols Shape the
Apparent Benefits of Hard-Sharing Multi-Task Learning in ADMET Prediction*.

The protocol was frozen before confirmatory computation on the leakage-controlled
design. This file, the pinned `requirements.txt`, and the git history of this
repository (commit timestamps) together constitute the immutable record. The
development pass (per-endpoint splits, subsequently identified as cross-task
identity exposure, plus a loss-scaling training bug) is excluded from
confirmatory statistics and reported separately as a contaminated stress test.

## 1. Data

- Therapeutics Data Commons binary ADMET endpoints (fixed set of eight):
  hERG, AMES, BBB_Martins, Pgp_Broccatelli, CYP2C9_Veith, CYP2D6_Veith,
  CYP3A4_Veith, Bioavailability_Ma.
- Canonicalization: RDKit canonical-SMILES round-trip
  (`MolFromSmiles`/`MolToSmiles`) under a fixed RDKit version (2026.03.4);
  rows failing canonicalization are dropped; rows sharing a canonical SMILES
  are deduplicated keeping the first occurrence (frozen rule, identical in
  development and confirmatory pipelines). Conflict statistics are reported
  in `results/data_audit.csv`.

## 2. Global molecule allocation (three partitions)

Each molecule is assigned to exactly one global partition; every endpoint
inherits that partition. No test molecule appears in any endpoint's training
or calibration partition (no cross-task identity exposure).

- Random protocol: molecule-level random allocation, 70% train / 10% cal / 20% test.
- Scaffold protocol: Bemis–Murcko scaffold-grouped allocation (molecules
  sharing a scaffold stay in one partition), 70/10/20 at the scaffold-group level.
- 5 split instances × 3 training seeds (both protocols).

## 3. Models

- ECFP4 (Morgan radius 2, 1024 bits) fingerprint MLP: shared trunk
  (256-ReLU-dropout0.2-128-ReLU) + per-endpoint linear heads.
- Hard-sharing MTL (task-balanced: 8 optimizer updates per task per epoch,
  batch 512, Adam lr 1e-3, weight decay 1e-4, 60 epochs) vs architecture-matched
  STL (same per-task update count).
- Sensitivity: three-layer GIN (hidden width 64, global add pool, per-endpoint
  linear heads, Adam lr 1e-3, batch 128, 80 epochs, 5 instances × 3 seeds,
  same leakage-controlled design).

## 4. Estimands

- Molecule-level log-loss contrast `d_i = NLL_STL,i - NLL_MTL,i` (positive favors MTL).
- Endpoint-macro mean Δ; protocol contrast `Γ = Δ_scaffold − Δ_random`.
- Secondary: AUROC/AUPRC contrasts, Brier reliability–resolution decomposition,
  calibration intercept/slope, continuous target-relative novelty
  (nearest-train Tanimoto, ECFP4, relative to the target endpoint's 70% train
  partition), endpoint-scale manipulation (downsampling of CYP endpoints).

## 5. Temperature calibration (strict per-model)

For each (split instance, training seed) run, protocol, model (MTL/STL), and
endpoint: one scalar temperature T is fitted by minimizing the mean log loss on
**that same trained model's own calibration partition** (disjoint from its
training and test partitions, T ∈ [0.2, 5.0], SciPy `minimize_scalar` bounded),
and applied to that same model's own test partition. No temperature is shared
across models.

## 6. Inference

Endpoints are a fixed target set; endpoint-level dispersion (paired t_7,
sign test, leave-one-endpoint-out) is reported as concordance/sensitivity for
these eight endpoints and does not imply an ADMET-endpoint population.
Molecular-level analyses cluster by standardized molecule (cluster bootstrap,
500–2000 resamples). No multiple-testing adjustment is applied to the primary
contrast; secondary analyses are labeled exploratory.

## 7. Reproducibility

- Environment pinned: Python 3.10, PyTorch 2.13.0 (CUDA 13.0 build),
  RDKit 2026.03.4, single NVIDIA RTX 4070 Ti SUPER.
- Split and seed lists, and their hashes, are recorded in `splits/`.
- All prediction tables and per-molecule contrasts are released under
  `results/`; analysis scripts under `analysis/`.

Frozen: 2026-08-03 (commit records the timestamp).
