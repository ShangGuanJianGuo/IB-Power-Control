import numpy as np
import numpy
import math
import cmath
import sys
import time
import functionRlocalscattering
import scipy.linalg as sl



def RandomAPLocations(L, K):
    posXY = np.random.uniform(low=100, high=900, size=(K, 2))
    UEXpositions = posXY[:, 0]
    UEYpositions = posXY[:, 1]
    UEpositions = UEXpositions + 1j * UEYpositions

    posXY = np.random.uniform(low=100, high=900, size=(L, 2))
    APXpositions = posXY[:, 0]
    APYpositions = posXY[:, 1]
    APpositions = APXpositions + 1j * APYpositions

    return APpositions, UEpositions


def functionCompute_SE_Dist(H_CF_lk, F_CF_lk_norm, prelogFactor, set_number, K):
    Q_B_K_K = np.sum(np.matmul(np.conj(H_CF_lk), F_CF_lk_norm.transpose((0, 1, 2, 4, 3))), axis=2)
    Q_B_K_K_1 = np.mean(Q_B_K_K, axis=1)
    Q_B_K_K_2 = Q_B_K_K_1.real ** 2 + Q_B_K_K_1.imag ** 2
    D_B_K = np.eye(K) * Q_B_K_K_2
    sumD = np.sum(D_B_K, axis=2)
    Q_B_K_K_3 = Q_B_K_K.real ** 2 + Q_B_K_K.imag ** 2
    Q_B_K_K_4 = np.mean(Q_B_K_K_3, axis=1)
    sumQ = np.sum(Q_B_K_K_4, axis=2)
    sinr = sumD / (sumQ - sumD + 1)
    SE_CF_MMSE = prelogFactor * np.log2(1.0 + sinr).reshape(set_number, K)
    sum_SE_CF_MMSE = np.sum(SE_CF_MMSE, axis=1)
    return sum_SE_CF_MMSE


def RandomChannel(L, K, M, set_number):
    # Pilot length
    tau_p = K
    # Select length of coherence block
    tau_c = 200
    # Compute the prelog factor
    prelogFactor = 1 - tau_p / tau_c
    # Communication bandwidth
    B = 20e6
    # Define noise figure at AP (in dB)
    noiseFigure = 7
    # Compute noise power
    noiseVariancedBm = -174 + 10 * np.log10(B) + noiseFigure
    # Pathloss exponent
    alpha = 36.7
    # Average channel gain in dB at a reference distance of 1 meter.
    constantTerm = -30.5
    # Prepare to store normalized spatial correlation matrices
    R = np.zeros((M, M, K, L), dtype='complex')
    # Angular standard deviation in the local scattering model (in degrees)
    ASDdeg = 10
    angletoUE = np.zeros((K, L))
    # Vertical distance between APs and UEs
    distanceVertical = 10

    # compute H
    H = np.zeros((set_number, nbrOfRealizations, L, K, M), dtype='complex')
    Beta = np.zeros((set_number, L, K), dtype='float64')
    start_time = time.time()
    for nbr in range(set_number):
        end_time = time.time()
        if nbr % 200 == 0:
            print(nbr, end_time - start_time)
            start_time = time.time()
        APpositions, UEpositions = RandomAPLocations(L, K)
        UEXpositions = UEpositions.real
        UEYpositions = UEpositions.imag
        APXpositions = APpositions.real
        APYpositions = APpositions.imag
        distances_agent = np.zeros((K, L))
        beta = np.zeros((K, L))
        for k in range(K):
            for j in range(L):
                distances_agent[k][j] = np.sqrt(distanceVertical ** 2 + abs(UEpositions[k] - APpositions[j]) ** 2)
                Xdist = UEXpositions[k] - APXpositions[j]
                Ydist = UEYpositions[k] - APYpositions[j]
                angletoUE[k, j] = np.angle(Xdist + 1j * Ydist)
                R[:, :, k, j] = functionRlocalscattering.R(M, angletoUE[k, j], ASDdeg)
            beta[k, :] = constantTerm - alpha * np.log10(distances_agent[k, :]) - noiseVariancedBm
        CH = np.sqrt(0.5) * (np.random.randn(M, nbrOfRealizations, K, L) + 1j * np.random.randn(M, nbrOfRealizations, K, L))
        CorrR = np.zeros((M, M, K, L), dtype='complex')
        betas = (10 ** (beta / 10))
        for j2 in range(L):
            for k2 in range(K):
                CorrR[:, :, k2, j2] = betas[k2, j2] * R[:, :, k2, j2]
                Rsqrt = sl.sqrtm(CorrR[:, :, k2, j2])
                H[nbr, 0, j2, k2, :] = np.matmul(Rsqrt, CH[:, :, k2, j2]).reshape(M)
        Beta[nbr] = betas.T
    W = math.sqrt(Pmax / K) * H / np.linalg.norm(H, axis=-1).reshape((set_number, nbrOfRealizations, L, K, 1))
    np.save("setHCF_L" + str(L) + "_K" + str(K) + "_M" + str(M) + "_number" + str(set_number) + ".npy", H)
    np.save("setWCF_L" + str(L) + "_K" + str(K) + "_M" + str(M) + "_number" + str(set_number) + ".npy", W)
    np.save("setBetaCF_L" + str(L) + "_K" + str(K) + "_M" + str(M) + "_number" + str(set_number) + ".npy", Beta)
    # MR
    sum_SE_CF_MMSE_MR = functionCompute_SE_Dist(H, W, 1, set_number, K)
    print(sum(sum_SE_CF_MMSE_MR) / set_number)
    print('generated')
    return H, W


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate correlated channels, MR precoders and true LSFC arrays')
    parser.add_argument('--L', type=int, default=40)
    parser.add_argument('--K', type=int, default=30)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--Pmax', type=float, default=1000.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.L, args.K, args.M, args.samples) < 1 or args.Pmax <= 0:
        parser.error('Dimensions, sample count and Pmax must be positive')
    Pmax = args.Pmax
    nbrOfRealizations = 1
    squareLength = 1000
    np.random.seed(args.seed)
    RandomChannel(args.L, args.K, args.M, args.samples)
