# Low-Overhead Distributed Power Control for Cell-Free Massive MIMO With Limited Fronthaul Capacity

Simulation source code associated with [the IEEE article](https://ieeexplore.ieee.org/abstract/document/11614834), DOI [10.1109/LWC.2026.3715345](https://doi.org/10.1109/LWC.2026.3715345).

This package contains the manuscript-related model, training and numerical experiments from `final_信息瓶颈_3.12`. It contains no plotting programs or plotting dependencies. Source files are flat at the repository root; channel arrays and checkpoints are generated locally. 中文逐文件用途见 [文件说明.txt](文件说明.txt).

## Installation

Use Python 3.11 and install the supplied dependencies. PyTorch automatically selects CUDA when available.

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root. Outputs are numeric CSV/NPY/MAT files and model checkpoints. Some comparison programs create runtime output directories.

## Channel data and main models

```bash
python CF_mMIMO_Channel.py --L 40 --K 30 --M 8 --samples 5000 --Pmax 1000 --seed 42
python train_federated_cluster.py --data_dir . --samples 5000 --train_samples 4000 --beta 0 --epochs 150
python train_and_evaluate_v2.py --data_dir . --samples 5000 --train_samples 4000 --beta_list 4 --epochs 100
python benchmark_upper_bound.py --data_dir . --samples 5000 --test_samples 500 --wmmse_iter 100
```

The generator saves H, W and Beta arrays. Repeat main training with beta 1, 2, 4, 8, 16, 32 and 64 as required, using100 epochs for those example settings. Main training writes `federated_cluster_beta_*.pth` and a training-history CSV. Centralized training writes `best_model_beta_*.pth` and `training_results_v2.npy`.

## Refinement-step experiment (Table II)

```bash
python evaluate_iterations.py --checkpoint best_model_beta_4.0.pth --data_dir . --samples 5000 --test_samples 500 --seed 42 --output_csv iteration_results.csv
```

This is a numeric evaluator, not a plotting script. CSV columns are step, average sum SE, absolute gain and percentage gain. The manuscript's centralized beta4 weight is distinct from the federated beta0 weight.

## Updated noisy-CSI experiments

```bash
python robustness_beta_sweep_corrected.py --data_dir . --checkpoint_dir . --samples 5000 --train_samples 4000 --betas 0,1,4,16,32,64 --noise_mode per_link --trials 5 --output_prefix robustness_IB
python wmmse_robustness_corrected.py --data_dir . --samples 5000 --train_samples 4000 --noise_mode per_link --trials 5 --iterations 100 --output_prefix robustness_WMMSE
```

These are the final noise-test entrypoints. They use the400-scene validation subset selected with seed42 from the first4000 scenarios, apply complex noise per AP-UE link, recompute MR from the observed noisy channel, and evaluate SE on the true channel. IB latent inference is deterministic by default. For all eight beta values, use `--betas 0,1,2,4,8,16,32,64`.

IB outputs `*_clean_audit.csv` and `*_results.csv`. `raw_noisy_se_mean/std` are direct noisy forward results. `anchored_se` is the stored historical clean score multiplied by the measured noisy/current-clean ratio; it is a separate normalization. Both fields are retained. WMMSE exports perfect CSI (internal SNR100) and the requested noisy-SNR means/stds.

## Payload and information-rate experiment

```bash
python measure_online_overhead.py --checkpoints federated_cluster_beta_0.0.pth federated_cluster_beta_64.0.pth --output overhead.csv
```

This exports current forward SE, checkpoint-stored SE, raw-CSI bits, uplink-latent bits, quantized-width bit accounting and the mean-refinement-round KL proxy. Fixed-width payload and the KL information-rate proxy are separate columns. No manually entered comparison SE percentages are included.

## Privacy experiment

```bash
python privacy_attack.py --data_dir . --checkpoint_dir . --samples 5000 --train_samples 4500 --epochs 500 --betas 0,1,2,4,8,16,32,64 --output_prefix privacy_attack
```

The decoder reconstructs normalized Beta-related vectors from latent messages. The program exports reconstruction MSE and cohort-mean Pearson to CSV/MAT. Keep the beta order when repeating a historical random-number sequence. Training all eight attack models can take considerable time.

## Comparison schemes (Table III) and precoder experiment

```bash
python wcl_baseline_suite.py --data_dir . --samples 5000 --eval_samples 300 --wmmse_samples 80 --output_dir wcl_results
python yu2023_cl_dpc_baseline.py --data_dir . --samples 5000 --max_samples 4000 --train_ratio 0.9 --legacy_torch_split --epochs 100 --early_stop_patience 20 --output_dir yu2023_results
python local_mmse_precoder_ablation.py --data_dir . --checkpoint federated_cluster_beta_0.0.pth --output_csv precoder_swap.csv
```

The baseline suite trains/evaluates LSFC local control, local neural control, deterministic no-KL federated control and centralized WMMSE. `ibfl_data.py` and `ibfl_models.py` are required dependencies. The cooperative scheme is the author's Yu2023-style adaptation. Numerical result files specify the evaluation method and subset. The precoder experiment reports direct and historical-normalized SE separately.

## Citation

Use `citation.bib` or GitHub's **Cite this repository** entry from `CITATION.cff` to cite the associated article.
