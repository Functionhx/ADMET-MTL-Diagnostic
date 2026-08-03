# ADMET-MTL-Diagnostic

**Protocol-Dependent, Temperature-Correctable Proper-Score Gains in Hard-Sharing Multi-Task ADMET Prediction**

A leakage-controlled diagnostic of task-balanced hard-sharing multi-task ADMET models: does the multi-task proper-score advantage depend on the evaluation protocol, endpoint scale, and chemical novelty?

## Key Findings

- Multi-task models assign lower log loss on all 8 endpoints under both protocols (grouped random and Bemis–Murcko scaffold allocation), with global molecule allocation controlling cross-task identity exposure.
- The protocol contrast Γ = Δ_scaffold − Δ_random = +0.10 (sign test p = 0.008; leave-one-endpoint-out range +0.07 to +0.11): the benefit is larger under the harder protocol.
- **Nested temperature calibration attenuates the advantage to ≈ 0** under both protocols and the interaction itself (calibrated Γ = +0.01): the gains reflect temperature-correctable differences in raw logit scale, not ranking information or a durable calibration advantage.
- Novelty dependence of the benefit reverses direction across protocols (declining with novelty under random, rising under scaffold).

## Repository Structure

```
ADMET-MTL-Diagnostic/
├── data/          # Data download scripts (TDC, versioned)
├── splits/        # Global molecule-partition manifests (random + scaffold, 5 instances) + hashes
├── models/        # b3_config.py, b3_main_experiment.py (R1: grouped random), b3_supplement.py (S1-S4: scaffold/downsampling/Brier)
├── calibration/   # Nested temperature calibration (per split/seed/protocol/model/endpoint)
├── analysis/      # R2/R2b (novelty, AUC/Brier decomposition), S3 (within-endpoint models), figures
├── results/       # Per-endpoint prediction tables and summary statistics
└── LICENSE
```

## Reproduction

1. `pip install rdkit scikit-learn torch pandas pyarrow`
2. Download TDC ADMET datasets (hERG, AMES, BBB_Martins, Pgp_Broccatelli, CYP2C9/2D6/3A4_Veith, Bioavailability_Ma) via `tdc.single_pred` into `data/`
3. Run models + analysis scripts in order; split manifests and seeds are fixed in `models/b3_config.py`

## Citing

(placeholder — citation info added at publication)

## License

MIT
