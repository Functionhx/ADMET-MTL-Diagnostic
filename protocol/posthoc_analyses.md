# Post Hoc Analyses (not in the confirmatory plan)

These analyses were added after the confirmatory results and are labeled
exploratory/diagnostic in the manuscript; they do not affect the primary
contrasts or their inference.

1. Novelty-matching reweighting (random protocol reweighted to the scaffold
   novelty distribution) — partition-geometry control (Results, Partition-geometry controls)
2. Hierarchical interaction model (protocol x novelty, endpoint fixed effects,
   molecule-clustered bootstrap) — same section
3. Label-permuted MTL control and pooled-pretrain + STL fine-tune (Results,
   Mechanistic controls; Supporting Information S5) — random protocol, 3 instances x 3 seeds
4. Top-k prioritization and asymmetric-cost utility (Results, Decision-level
   implications; Supporting Information S6)
5. Layered bootstrap uncertainty decomposition (molecule / split-instance block /
   endpoint-resample) — added when the molecule-only interval was judged too narrow
6. Data-curation sensitivity (13 conflicting molecules excluded) — Methods
