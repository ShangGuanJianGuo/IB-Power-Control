# Low-Overhead Distributed Power Control for Cell-Free Massive MIMO With Limited Fronthaul Capacity

Author source snapshot for [the IEEE article](https://ieeexplore.ieee.org/abstract/document/11614834), DOI [10.1109/LWC.2026.3715345](https://doi.org/10.1109/LWC.2026.3715345).

This revision replaces the earlier eight-script upload with the manuscript-stage sources from `final_信息瓶颈_3.12`, including the **updated per-link noisy-CSI evaluators**. It is a source-only release: no channel arrays, checkpoints, manuscript PDFs or historical output tables are included. Every repository file is at the root.

**Status:** several saved numerical results have been replayed, but known physical-power, LSFC-input and historical-checkpoint mismatches remain. This is a documented final-source snapshot, not a claim that a fresh run reproduces every published number. Read [KNOWN_ISSUES.md](KNOWN_ISSUES.md) before interpreting SE. 中文逐文件说明见 [文件说明.txt](文件说明.txt).

## Environment

Python 3.11; the checked versions are pinned in `requirements.txt`. CUDA is optional; PyTorch selects a GPU when available.

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root. Supply your own compatible historical arrays/checkpoints for a historical replay, or generate and train new ones below. Newly drawn channels are not the historical dataset.

## 1. Generate local inputs

```bash
python CF_mMIMO_Channel.py --L 40 --K 30 --M 8 --samples 5000 --Pmax 1000 --seed 42
```

Outputs: `setHCF_L40_K30_M8_number5000.npy`, `setWCF_L40_K30_M8_number5000.npy`, and `setBetaCF_L40_K30_M8_number5000.npy`. H/W together occupy about 1.54 GB. The generator retains the author's MR amplitude convention; it has not been silently replaced with a physically revised experiment. Its CLI and SciPy compatibility are packaging adaptations.

New Beta files make the newer baseline helpers use actual LSFC. The supplied historical experiment data lacked these files and used instantaneous-power fallback. These are different input protocols. The legacy main training loader still estimates Beta from H; see the issue notes.

## 2. Train the legacy main architecture

```bash
python train_federated_cluster.py --mode train --data_dir . --samples 5000 --train_samples 4000 --beta 0 --epochs 150
python train_federated_cluster.py --mode train --data_dir . --samples 5000 --train_samples 4000 --beta 1 --epochs 100
```

Repeat with beta 2, 4, 8, 16, 32 and 64 as needed. Outputs are `federated_cluster_beta_*.pth`. These commands exercise the original joint-data simulation with cluster labels; they do not establish a deployed data-local FL protocol. All new main weights must be trained with the same forward definition used at evaluation.

The centralized checkpoint used for the published Table II is a separate model output:

```bash
python train_and_evaluate_v2.py --data_dir . --samples 5000 --train_samples 4000 --beta_list 4 --epochs 100
python plot_iteration_se_v2.py --checkpoint best_model_beta_4.0.pth --data_dir . --samples 5000 --test_samples 500 --seed 42
```

With the author's historical centralized beta-4 checkpoint and original arrays, the step means replay as 45.470445, 82.739917, 114.126454 and 115.699770 bit/s/Hz. Relative improvements are 81.96%, 37.93%, 1.38%. Fresh training is not guaranteed to land on that checkpoint.

For the federated validation trace instead:

```bash
python plot_iteration_se_federated.py --checkpoint federated_cluster_beta_0.0.pth
```

It is a different checkpoint and split, not a substitute for Table II.

## 3. WMMSE reference

```bash
python benchmark_upper_bound.py --data_dir . --samples 5000 --test_samples 500 --wmmse_iter 100
```

This standalone routine selects the first requested scenarios. It is not automatically paired with validation/tail experiments. Its original filename is retained; WMMSE is a reference algorithm, not a certified global optimum.

## 4. FINAL noisy-CSI tests

Use these updated scripts, rather than the original whole-scenario-noise implementations:

```bash
python robustness_beta_sweep_corrected.py --data_dir . --checkpoint_dir . --samples 5000 --train_samples 4000 --betas 0,1,4,16,32,64 --noise_mode per_link --trials 5 --plot_value raw --output_prefix robustness_final
python wmmse_robustness_corrected.py --data_dir . --samples 5000 --train_samples 4000 --noise_mode per_link --trials 5 --iterations 100 --output_prefix wmmse_final
python plot_paper_results.py robustness --ib_csv robustness_final_results.csv --wmmse_csv wmmse_final_results.csv --value raw --output robustness_raw.png
```

For the full eight-beta sweep use `--betas 0,1,2,4,8,16,32,64`. The old filenames `test_noise_federated.py` and `test_wmmse_noise.py` now forward to the corrected entrypoints and accept their arguments. Legacy `--test_samples` options are not used by this validation-split protocol; use `--max_eval_samples` only for an explicit shortened check.

The corrected protocol uses the same 400 validation scenes drawn by `torch.randperm(seed=42)` from the first 4000 scenarios. It adds complex noise per AP–UE link, recomputes MR from the noisy estimate, and measures SE on the clean channel with the used precoder. Default noisy SNRs: 20, 15, 10, 5, 0, -5 dB. WMMSE also exports the perfect-CSI row (internal SNR 100). Latent inference is deterministic by default in this evaluator.

IB exports `*_clean_audit.csv` and `*_results.csv`, containing:

- `checkpoint_clean_se`: the checkpoint's stored historical score;
- `raw_current_clean_se`: current-code clean forward SE;
- `raw_noisy_se_mean`, `raw_noisy_se_std`: direct noisy forward results;
- `anchored_se`: `checkpoint_clean_se * min(raw_noisy_se_mean / raw_current_clean_se, 1)` by default.

Anchored SE is an explicit historical normalization. It is not the direct SE and does not repair power feasibility. The author's corrected single-script plot defaults to anchored values, with the axis labeled accordingly; the commands above explicitly choose raw plots. To inspect the historical-normalized comparison, use `--value anchored` with `plot_paper_results.py`.

On the author's historical inputs, all 36 saved IB mean points and all seven saved WMMSE SNR means replayed (IB maximum difference about 2.84e-14). This validates that saved CSV workflow, not every modeling assumption.

## 5. Other comparison experiments

```bash
python wcl_baseline_suite.py --data_dir . --samples 5000 --eval_samples 300 --wmmse_samples 80 --output_dir wcl_results
python yu2023_cl_dpc_baseline.py --data_dir . --samples 5000 --max_samples 4000 --train_ratio 0.9 --legacy_torch_split --epochs 100 --early_stop_patience 20 --output_dir yu2023_results
```

The first script trains local DNN and deterministic no-KL FL comparison models. Its optional `--proposed_checkpoint` expects the newer `ibfl_models.py` checkpoint schema; old `federated_cluster_beta_*.pth` files cannot be substituted. The supplied final folder did not contain that optional main checkpoint. Missing optional models are reported rather than assigned invented results. Runtime outputs may create directories; **the distributed source package itself is flat**.

The Yu2023 code is the author's adaptation, not an official upstream implementation. Historical local DNN, no-KL FL and cooperative checkpoints replayed as 108.189857, 93.298347 and 133.936218 bit/s/Hz on their respective original subsets. Dividing by the historical rounded 136.84 reference reconstructs 79.06%, 68.18%, 97.88%; the cohorts are not identical paired tests.

## 6. Information-rate / payload accounting and precoder swap

```bash
python measure_online_overhead.py --checkpoints federated_cluster_beta_0.0.pth federated_cluster_beta_64.0.pth --output overhead_measurement.csv
python local_mmse_precoder_ablation.py --data_dir . --checkpoint federated_cluster_beta_0.0.pth --output_csv precoder_swap.csv
```

The overhead script retains historical comparison ratios and marks their source. It also exports direct current SE, stored checkpoint SE, KL bits and fixed-width payload separately. At the checked historical beta-0 model, the KL proxy is 4.6777% of raw CSI; three-round FP32 uplink latents are 10.00%, before downlink/control/header costs. These are different definitions. No network-wire compression claim is made. The precoder swap exports both raw and historical-normalized results.

## 7. Latent reconstruction attack

```bash
python privacy_attack.py --data_dir . --checkpoint_dir . --samples 5000 --train_samples 4500 --epochs 500 --betas 0,1,2,4,8,16,32,64 --output_prefix privacy_attack
python plot_paper_results.py privacy --mat privacy_attack_data.mat --output privacy_reconstruction.png
```

This is the final `privacy_attack copy 3.py` weighted supervised decoder attack. It maps intercepted z to normalized Beta-related vectors; it does not implement gradient inversion. MAT/CSV now save the test-pool mean Pearson, which the author script previously only printed. A single-example Pearson is not substituted if that field is missing.

Retain the complete beta order for historical RNG replay. Changing beta selection changes later random-number history. In the audit, beta 0 (500 attack epochs) exactly replayed MSE 1.076514 and mean Pearson 0.2817; the entire eight-point attack curve was not retrained. The script preserves native batching and can take considerable time, particularly with CUDA tensors indexed sample by sample.

## File guide and validation

See [文件说明.txt](文件说明.txt) for every file's purpose, [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) for source origins/adaptations, and [VALIDATION.md](VALIDATION.md) for this release's checks. No Word/test-report code is included.

## Citation

Use `citation.bib` or GitHub's **Cite this repository**, which reads `CITATION.cff`. Please cite the associated IEEE article. This repository does not add a new license grant for the author's sources.
