# Release validation — 2026-11-25

## This packaged revision

- All20 Python files parsed successfully.
- Computation ASTs for8 unadapted main/model/baseline files match the final manuscript sources.
- An isolated copy passed17 stages: channel/H/W/Beta generation; one-epoch main and centralized training; standalone WMMSE; both canonical corrected noise scripts; both old-name forwarding entrypoints; combined noise plotting; central/federated iteration tracing; overhead export; precoder swap; a2-epoch privacy attack and mean-Pearson MAT/CSV export; privacy plotting; local/no-KL FL comparison training; cooperative baseline training.
- The two old-name noise entrypoints produced byte-identical numeric CSVs to the canonical new entrypoints on the small check.
- Small network: L4,K3,M2,Pmax10,20 generated scenarios,16 training-pool scenarios. This is a functional check, not publication-performance validation.
- Current package on original400-scene validation inputs: beta0 and32, SNR20 and0dB,5 trials; maximum direct-SE error vs saved corrected CSV=0, anchored error=0.
- Current WMMSE package on the same400 validation scenes: perfect CSI,20 and0dB; maximum mean error vs saved CSV=0.
- New result-driven plots were visually inspected. Small-check images are not paper results and are not included in the source release.

## Prior full-data audit of the same final computation sources

Eight federated checkpoints were each evaluated on400 validation scenes with3 inference seeds and1000 held-out scenes. All36 saved corrected IB noise mean points and7 WMMSE mean points were replayed. The centralized beta4 checkpoint reproduced Table II's81.96%,37.93%,1.38% gains. Three comparison-model percentages and the4.68% KL proxy were reconstructed under their historical definitions. The privacy beta0 endpoint matched MSE1.076514/Pearson0.2817; the full eight-attack sequence was not retrained.

No new100/150-epoch converged training or physically revised end-to-end qualification is claimed. See KNOWN_ISSUES.md: power conventions, LSFC fallback, input/checkpoint provenance and information-rate definitions still matter. No channel arrays, checkpoints or historical result CSVs are distributed.
