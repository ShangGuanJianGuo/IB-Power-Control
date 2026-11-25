# Low-Overhead Distributed Power Control for Cell-Free Massive MIMO With Limited Fronthaul Capacity

Simulation source code accompanying the article by **Yukun Ma, Jiayi Zhang, Ziheng Liu, Huahua Xiao, Xiaodan Zhang, Derrick Wing Kwan Ng, and Bo Ai**, published in *IEEE Wireless Communications Letters*, vol. 15, pp. 4225–4229, 2026.

- [IEEE Xplore article](https://ieeexplore.ieee.org/abstract/document/11614834)
- [DOI: 10.1109/LWC.2026.3715345](https://doi.org/10.1109/LWC.2026.3715345)

## Overview

The implementation contains an information-bottleneck (IB) iterative power-control network, user-cluster federated training, centralized IB training, a WMMSE baseline, and imperfect-CSI robustness experiments. The channel generator creates spatially correlated cell-free massive MIMO channels and MR precoders.

This is a **source-only package**. Generate the channel arrays locally, train the models, and then run the evaluations. Existing channel files, pretrained weights and historical result files are not bundled. All repository files are placed at the root level.

## Environment and installation

Checked environment: Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1, PyTorch 2.9.1, Matplotlib 3.11.1 and tqdm 4.66.5. Version pins are provided in `requirements.txt`. PyTorch scripts automatically select CUDA when available, otherwise CPU.

```bash
python -m pip install -r requirements.txt
```

Run the following commands **from the repository root**. Data, checkpoints and figures are read from or written to that directory.

## Core source files

| File | Purpose |
| --- | --- |
| `CF_mMIMO_Channel.py` | Generate AP/UE locations, correlated channels and MR precoders. |
| `functionRlocalscattering.py` | Construct the local-scattering spatial correlation matrix. |
| `model_architecture_v2.py` | IB encoder, attention aggregation, iterative power policy and SE loss. |
| `train_federated_cluster.py` | User-cluster federated training, validation and checkpoint saving. |
| `train_and_evaluate_v2.py` | Centralized IB training, validation and beta comparisons. |
| `benchmark_upper_bound.py` | Original WMMSE power-control baseline implementation. |
| `test_noise_federated.py` | Noisy-CSI evaluation of trained federated checkpoints. |
| `test_wmmse_noise.py` | Noisy-CSI WMMSE evaluation with recomputed MR precoders. |

## 1. Generate channel data

```bash
python CF_mMIMO_Channel.py --L 40 --K 30 --M 8 --samples 5000 --Pmax 1000 --seed 42
```

This creates:

```text
setHCF_L40_K30_M8_number5000.npy
setWCF_L40_K30_M8_number5000.npy
```

Defaults: 40 APs, 30 users, 8 antennas per AP, one channel realization per scenario, a 1000 m x 1000 m deployment region, and a per-AP transmit-power parameter of 1000 mW. AP and user coordinates are sampled in the central [100, 900] m interval on both axes, as in the supplied generator. The generator uses a 20 MHz bandwidth and a 7 dB receiver noise figure to normalize channel gains. Training and rate routines use normalized noise power 1.

The two 5000-scenario arrays occupy approximately 1.54 GB in total. They are generated locally and ignored by Git.

## 2. Train the federated IB model

```bash
python train_federated_cluster.py --mode train --data_dir . --samples 5000 --train_samples 4000 --L 40 --K 30 --M 8 --beta 8 --epochs 100
```

The example produces `federated_cluster_beta_8.0.pth`. To compare multiple IB weights, run the same command again with a different `--beta` value. The robustness evaluator automatically finds files matching `federated_cluster_beta_*.pth` in the current directory.

The original script also retains its `--mode unlearn` entry point. The primary power-control and robustness workflow above does not require that optional mode.

## 3. Train the centralized IB comparison

```bash
python train_and_evaluate_v2.py --data_dir . --samples 5000 --train_samples 4000 --L 40 --K 30 --M 8 --beta_list 10 --epochs 200
```

Multiple values are accepted, for example `--beta_list 0 1 10`. This script saves `best_model_beta_*.pth`, training histories and figures. These centralized checkpoints are a different output format from the federated checkpoints used by `test_noise_federated.py`.

## 4. Evaluate the standalone WMMSE baseline

```bash
python benchmark_upper_bound.py --data_dir . --samples 5000 --test_samples 500 --wmmse_iter 100
```

This writes `wmmse_baseline_L40_K30_M8.npy`, containing per-scenario sum SE and aggregate statistics. The filename `benchmark_upper_bound.py` is retained from the original code; it denotes the supplied WMMSE baseline implementation.

## 5. Compare noisy-CSI robustness

Run this after generating data and training at least one federated checkpoint:

```bash
python test_noise_federated.py --data_dir . --samples 5000 --test_samples 500
python test_wmmse_noise.py --data_dir . --samples 5000 --test_samples 500 --iterations 100
```

Both scripts evaluate the final 500 scenarios with these explicit arguments. Tested SNR levels are 20, 15, 10, 5, 0 and -5 dB, plus the perfect-CSI case represented internally by 100. Outputs include robustness figures; the WMMSE routine also writes its detailed CSV results.

### Data selection and interpretation

- Training uses the first `--train_samples` scenarios with an internal 90%/10% training/validation split.
- The noisy-CSI evaluators select the final `--test_samples` scenarios. With 5000 scenarios, a 4000-scenario training pool and 500 evaluation scenarios, these evaluation channels are outside the training pool.
- The standalone WMMSE baseline selects the first `--test_samples` scenarios. Its aggregate value is therefore not a paired result on the robustness evaluators' tail subset.
- Rates are simulation spectral efficiencies. The IB regularizer is a variational information-rate proxy, not a serialized network-byte measurement.
- Parameter examples above are runnable settings; newly generated channels and newly trained weights can give different numerical results from historical paper figures.

For the available options:

```bash
python CF_mMIMO_Channel.py --help
python train_federated_cluster.py --help
python train_and_evaluate_v2.py --help
python benchmark_upper_bound.py --help
python test_noise_federated.py --help
python test_wmmse_noise.py --help
```

## Packaging and compatibility notes

The eight core files were selected by comparing the author's `11.21` source collection with the corresponding Python files in `信息瓶颈_11.21_final`. This package retains the latter Python defaults and the original power-control, training and baseline computations. Packaging changes are:

1. Rename `1_CF_mMIMO_Channel.py` to `CF_mMIMO_Channel.py`; expose generator parameters and a seed, with 5000 scenarios by default to match the training filenames.
2. Flatten the Toeplitz input explicitly in the scattering function for compatibility with newer SciPy.
3. Load federated checkpoints onto CPU before copying weights into the selected model device, improving checkpoint portability.
4. Remove the fixed historical `136.8` WMMSE reference lines from centralized-training figures. Baseline values are obtained by running the baseline program.

Validation performed for this package: source syntax and CLI entry checks, channel generation, one-epoch federated and centralized training, WMMSE evaluation, and both noisy-CSI evaluations on a small 10-scenario, 4-AP, 3-user, 2-antenna configuration. This is a functional check, not a rerun of the full published experiments.

## Citation

Please cite the article when using this code:

```bibtex
@article{Ma2026LowOverhead,
  author  = {Ma, Yukun and Zhang, Jiayi and Liu, Ziheng and Xiao, Huahua and Zhang, Xiaodan and Ng, Derrick Wing Kwan and Ai, Bo},
  title   = {Low-Overhead Distributed Power Control for Cell-Free Massive {MIMO} With Limited Fronthaul Capacity},
  journal = {IEEE Wireless Communications Letters},
  year    = {2026},
  volume  = {15},
  pages   = {4225--4229},
  doi     = {10.1109/LWC.2026.3715345},
  url     = {https://ieeexplore.ieee.org/abstract/document/11614834}
}
```

The same reference is provided in `citation.bib` and `CITATION.cff`. 中文文件用途见 `文件说明.txt`；仓库管理员上传步骤见 `上传说明.txt`。
