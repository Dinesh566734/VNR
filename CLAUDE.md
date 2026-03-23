# Sentinel-UPI Memory

- Project root lives at `sentinel-upi/` under the workspace container.
- Hyperparameters and loader adaptation knobs are defined in `config/hparams.yaml`.
- Phase 1 focuses on the PaySim-to-UPI loader; later modules are scaffolded as placeholders.
- For P2P edges, `merchant_type=peer_to_peer` and `mcc_weight=1.0` are treated as the neutral baseline.
- If source PaySim class imbalance falls outside the target 0.10%-0.20% window, the loader may recalibrate labels toward the configured target while preserving deterministic behavior.
