"""Plot newly exported robustness/privacy results without manual point entry."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='kind', required=True)
    noise = sub.add_parser('robustness')
    noise.add_argument('--ib_csv', required=True)
    noise.add_argument('--wmmse_csv', required=True)
    noise.add_argument('--value', choices=['raw', 'anchored'], default='raw')
    noise.add_argument('--output', default='robustness_comparison.png')
    privacy = sub.add_parser('privacy')
    privacy.add_argument('--mat', default='privacy_attack_data.mat')
    privacy.add_argument('--output', default='privacy_reconstruction.png')
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 12, 'axes.linewidth': .8, 'savefig.dpi': 300})
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if args.kind == 'robustness':
        ib = read_csv(args.ib_csv)
        wm = read_csv(args.wmmse_csv)
        clean = [float(r['sum_se_mean']) for r in wm if float(r['snr_db']) >= 100]
        if len(clean) != 1 or clean[0] <= 0:
            raise ValueError('WMMSE CSV must include exactly one positive perfect-CSI reference')
        ref = clean[0]
        points = sorted([r for r in wm if float(r['snr_db']) < 100], key=lambda r: float(r['snr_db']))
        ax.plot([float(r['snr_db']) for r in points], [100 * float(r['sum_se_mean']) / ref for r in points],
                'k--p', lw=1.8, ms=6, label='WMMSE')
        key = 'anchored_se' if args.value == 'anchored' else 'raw_noisy_se_mean'
        for i, beta in enumerate(sorted({float(r['beta']) for r in ib})):
            rows = sorted([r for r in ib if float(r['beta']) == beta], key=lambda r: float(r['snr_db']))
            ax.plot([float(r['snr_db']) for r in rows], [100 * float(r[key]) / ref for r in rows],
                    marker=['o', 's', '^', 'D', 'v', '>', '<', 'x'][i % 8], lw=1.5, ms=5,
                    label=rf'IB, $\beta={beta:g}$')
        ax.set_xlabel('Channel estimation SNR (dB)')
        ax.set_ylabel('SE / perfect-CSI WMMSE (%)')
        if args.value == 'anchored':
            ax.set_title('IB curves use historical-clean anchoring', fontsize=10)
        ax.legend(fontsize=9, ncol=2)
    else:
        data = loadmat(args.mat)
        if 'mean_pearson_values' not in data:
            raise ValueError('MAT lacks cohort-mean Pearson; rerun the updated privacy_attack.py. Example-vector correlation is not a substitute.')
        beta = data['beta_values'].ravel()
        mse = data['mse_values'].ravel()
        corr = data['mean_pearson_values'].ravel()
        if not len(beta) or len(beta) != len(mse) or len(beta) != len(corr):
            raise ValueError('Beta, MSE and mean Pearson arrays must have matching nonzero lengths')
        order = np.argsort(beta)
        positions = np.arange(len(beta))
        ax.plot(positions, mse[order], 'bs-', lw=1.7, label='Reconstruction MSE')
        right = ax.twinx()
        right.plot(positions, corr[order], 'ro-', lw=1.7, label='Mean Pearson correlation')
        ax.set_xticks(positions, [f'{b:g}' for b in beta[order]])
        ax.set_xlabel('IB penalty beta')
        ax.set_ylabel('Reconstruction MSE', color='b')
        right.set_ylabel('Mean Pearson correlation', color='r')
        lines = ax.get_lines() + right.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], fontsize=9)
    ax.grid(alpha=.25)
    ax.tick_params(direction='in', top=True, right=args.kind != 'privacy')
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f'Saved {Path(args.output)}')


if __name__ == '__main__':
    main()
