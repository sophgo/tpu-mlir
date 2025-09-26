#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import numpy as np
import math
from scipy import spatial

try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cp = None
    cupy_available = False


class CaliMathCpu:

    def prepare(self, array):
        return array

    def sort_distr(self, array, length):

        def first_k(a, k):
            a_sort = np.sort(a)
            return a_sort[-k:]

        return first_k(array, length)

    def sort_distr_percentile(self, array, length):

        def first_k(a, k):
            a_sort = np.sort(a)
            f = a_sort[:k].copy()
            r = a_sort[-k:].copy()
            a = np.concatenate((f, r))
            return a

        return first_k(array, length)

    def get_percentile(self, array, percentile_len: int = 1):
        return self.sort_distr_percentile(array.flatten(), percentile_len)

    def get_kurtosis(signal):
        """Calculates the kurtosis of a given signal.
        """
        signal = signal.flatten()
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return float('inf')

        standardized_signal = (signal - mean) / (std + 1e-8)
        kurtosis = np.mean(standardized_signal**4)

        return kurtosis.item()

    def histogram(self, ndarray, abs_max, bin_num: int = 2048):
        t = np.abs(ndarray.flatten())
        t = t[t != 0]
        width = abs_max / (bin_num - 1)
        if t.size > 0:
            hist, _ = np.histogram(np.floor(t / width + 0.5),
                                   bins=bin_num,
                                   range=(0, bin_num - 1),
                                   density=False)
        else:
            hist = np.zeros(bin_num)
        hist = hist.astype(np.int32)
        return hist, width

    def combine_histogram(self, old_hist, arr, new_min, new_max, new_th, bin_num: int = 2048):
        """ Collect layer histogram for arr and combine it with old histogram.
        """
        (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist
        if new_th <= old_th:
            hist, width = self.histogram(arr, old_th, len(old_hist))
            return (old_hist + hist, old_hist_edges, min(old_min, new_min), max(old_max,
                                                                                new_max), old_th)
        else:
            # Need to generate new histogram with new_th
            old_num_bins = len(old_hist)
            old_step = old_th / old_num_bins
            half_increased_bins = int((new_th - old_th) // old_step + 1)
            new_num_bins = half_increased_bins + old_num_bins
            if new_num_bins > 8000000:
                warnings.warn("校准图片差异过大,如果方便请调整校准图片顺序或者替换校准图片", UserWarning)
                hist, hist_edges = self.histogram(arr, new_th, bin_num)
                return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_th)
            new_th = half_increased_bins * old_step + old_th
            hist, hist_edges = self.histogram(arr, new_th, new_num_bins)
            hist[0:new_num_bins - half_increased_bins] += old_hist
            return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_th)

    def threshold_mse(self, array, bits: list = [8]):
        if np.abs(np.min(array) - 0) < 1e-6:
            unsigned = 4
        else:
            unsigned = 1
        abs_x = np.abs(array.flatten())
        thresholds = {}
        for bit in bits:
            s_n = abs_x.sum() / abs_x[abs_x > 0].size
            for _ in range(20):
                s_n_plus_1 = abs_x[abs_x > s_n].sum() / (
                    1 /
                    (4**bit) / 3 / unsigned * abs_x[abs_x <= s_n].size + abs_x[abs_x > s_n].size)
                if np.abs(s_n_plus_1 - s_n) < 1e-6:
                    break
                s_n = s_n_plus_1
            thresholds[bit] = s_n
        return thresholds

    def threshold_aciq_gauss(self, array, bits: list = [8]):
        thresholds = {}
        for bit in bits:
            alpha = 3.92403337 if bit == 8 else 2.55913642
            gaussian_const = (0.5 * 0.35) * (1 + (math.pi * math.log(4))**0.5)
            N = array.size
            std = ((np.max(array) - np.min(array)) * gaussian_const) / ((2 * math.log(N))**0.5)
            thresholds[bit] = alpha * std
        return thresholds

    def threshold_aciq_laplace(self, array, bits: list = [8]):
        thresholds = {}
        for bit in bits:
            beta = 9.89675982 if bit == 8 else 5.02864014
            b = np.mean(abs(array - np.mean(array)))
            thresholds[bit] = beta * b
        return thresholds


class CaliMathCuda:

    def prepare(self, array):
        return cp.asarray(array)

    def sort_distr(self, array, length):

        def first_k(a, k):
            a_sort = cp.sort(a)
            return a_sort[-k:]

        return first_k(array, length)

    def sort_distr_percentile(self, array, length):

        def first_k(a, k):
            a_sort = cp.sort(a)
            f = a_sort[:k].copy()
            r = a_sort[-k:].copy()
            a = cp.concatenate((f, r))
            return a

        return first_k(array, length)

    def get_percentile(self, array, percentile_len: int = 1):
        return self.sort_distr_percentile(array.flatten(), percentile_len)

    def get_kurtosis(signal):
        """Calculates the kurtosis of a given signal.
        """
        signal = signal.flatten()
        mean = cp.mean(signal)
        std = cp.std(signal)
        if std == 0:
            return float('inf')

        standardized_signal = (signal - mean) / (std + 1e-8)
        kurtosis = cp.mean(standardized_signal**4)

        return kurtosis.item()

    def histogram(self, ndarray, abs_max, bin_num: int = 2048):
        t = cp.abs(ndarray.flatten())
        t = t[t != 0]
        width = abs_max / (bin_num - 1)
        if t.size > 0:
            hist, _ = cp.histogram(cp.floor(t / width + 0.5),
                                   bins=bin_num,
                                   range=(0, bin_num - 1),
                                   density=False)
        else:
            hist = cp.zeros(bin_num)
        hist = hist.astype(cp.int32)
        return hist, width

    def combine_histogram(self, old_hist, arr, new_min, new_max, new_th, bin_num: int = 2048):
        """ Collect layer histogram for arr and combine it with old histogram.
        """
        (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist
        if new_th <= old_th:
            hist, width = self.histogram(arr, old_th, len(old_hist))
            return (old_hist + hist, old_hist_edges, min(old_min, new_min), max(old_max,
                                                                                new_max), old_th)
        else:
            # Need to generate new histogram with new_th
            old_num_bins = len(old_hist)
            old_step = old_th / old_num_bins
            half_increased_bins = int((new_th - old_th) // old_step + 1)
            new_num_bins = half_increased_bins + old_num_bins
            if new_num_bins > 8000000:
                warnings.warn("校准图片差异过大,如果方便请调整校准图片顺序或者替换校准图片", UserWarning)
                hist, hist_edges = self.histogram(arr, new_th, bin_num)
                return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_th)
            new_th = half_increased_bins * old_step + old_th
            hist, hist_edges = self.histogram(arr, new_th, new_num_bins)
            hist[0:new_num_bins - half_increased_bins] += old_hist
            return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_th)

    def threshold_mse(self, array, bits: list = [8]):
        if cp.abs(cp.min(array) - 0) < 1e-6:
            unsigned = 4
        else:
            unsigned = 1
        abs_x = cp.abs(array.flatten())
        thresholds = {}
        for bit in bits:
            s_n = abs_x.sum() / abs_x[abs_x > 0].size
            for _ in range(20):
                s_n_plus_1 = abs_x[abs_x > s_n].sum() / (
                    1 /
                    (4**bit) / 3 / unsigned * abs_x[abs_x <= s_n].size + abs_x[abs_x > s_n].size)
                if cp.abs(s_n_plus_1 - s_n) < 1e-6:
                    break
                s_n = s_n_plus_1
            thresholds[bit] = s_n
        return thresholds

    def threshold_aciq_gauss(self, array, bits: list = [8]):
        thresholds = {}
        for bit in bits:
            alpha = 3.92403337 if bit == 8 else 2.55913642
            gaussian_const = (0.5 * 0.35) * (1 + (math.pi * math.log(4))**0.5)
            N = array.size
            std = ((cp.max(array) - cp.min(array)) * gaussian_const) / ((2 * math.log(N))**0.5)
            thresholds[bit] = alpha * std
        return thresholds

    def threshold_aciq_laplace(self, array, bits: list = [8]):
        thresholds = {}
        for bit in bits:
            beta = 9.89675982 if bit == 8 else 5.02864014
            b = cp.mean(abs(array - cp.mean(array)))
            thresholds[bit] = beta * b
        return thresholds


def cosine_sim(x, y):
    x[np.isnan(x)] = 0.0
    y[np.isnan(y)] = 0.0
    cosine_similarity = 1 - spatial.distance.cosine(x.flatten().astype(np.float32),
                                                    y.flatten().astype(np.float32))
    return cosine_similarity


if cupy_available:
    math_impl = CaliMathCuda()
    POOL_THREADS = 1
else:
    math_impl = CaliMathCpu()
    POOL_THREADS = 6
