# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================


import numpy as np
import sys
import struct
# from math import fabs
from enum import IntEnum
from math import *
from collections import OrderedDict


def second(elem):
    return elem[1]


def get_topk(a, k):
    k = min(a.size, k)
    idx = np.argpartition(-a.ravel(), k - 1)[:k]
    # return np.column_stack(np.unravel_index(idx, a.shape))
    topk = list(zip(idx, np.take(a, idx)))
    #return topk
    topk.sort(key=second, reverse=True)
    return topk


class TensorCompare():
    NOT_MATCH = "NOT_MATCH"
    EQUAL = "EQUAL"
    NOT_EQUAL = "NOT_EQUAL"
    CLOSE = "CLOSE"
    SIMILAR = "SIMILAR"
    NOT_SIMILAR = "NOT_SIMLIAR"

    def __init__(self,
                 close_order_tol=3,
                 cosine_similarity_tol=0.99,
                 euclidean_similarity_tol=0.90,
                 signal_to_quantization_noise_tol=50,
                 per_axis_compare=-1):
        self.close_order_tol = close_order_tol
        self.cosine_similarity_tol = cosine_similarity_tol
        self.euclidean_similarity_tol = euclidean_similarity_tol
        self.signal_to_quantization_noise_tol = signal_to_quantization_noise_tol
        self.per_axis_compare = per_axis_compare
        return

    def square_rooted(self, x):
        return sqrt(np.sum(np.power(x, 2)))

    def cosine_similarity(self, x, y):
        numerator = np.sum(x, y)
        denominator = self.square_rooted(x) * self.square_rooted(y)
        return round(numerator / float(denominator), 3)

    def euclidean_distance(self, x, y):
        return sqrt(np.sum(np.power(x - y, 2)))

    def sqnr_similarity(self, signal_raw, signal_dequant):
        # SQNR is non-commutative
        # Unlike other distance function
        # Cannot change the order of signal_raw and signal_dequant
        raw = signal_raw.ravel()
        dequant = signal_dequant.ravel()

        noise = raw - dequant

        avg_raw = np.sum(raw) / raw.size
        avg_noise = np.sum(noise) / noise.size

        var_raw_zero_mean = np.sum(np.square(raw - avg_raw))
        var_noise_zero_mean = np.sum(np.square(noise - avg_noise))
        if var_noise_zero_mean == 0 or var_raw_zero_mean == 0:
            return float('inf')
        sqnr = 10 * np.log10(var_raw_zero_mean / var_noise_zero_mean)

        return sqnr

    def all_diffs(self, d1, d2):
        diffs = list()
        d1f = d1.ravel()
        d2f = d2.ravel()
        if d1f.dtype == np.int8:
            assert (d2f.dtype == np.int8)
            for i in range(len(d1f)):
                if (d1f[i] != d2f[i]):
                    diffs.append((i, d1f[i], d2f[i]))
        else:
            atol = 10**(-self.close_order_tol)
            rtol = 10**(-self.close_order_tol)
            for i in range(len(d1f)):
                if fabs(d1f[i] - d2f[i]) > (atol + rtol * fabs(d2f[i])):
                    diffs.append((i, d1f[i], d2f[i]))
            return diffs

    def diff_details(self, d1, d2, verbose):
        details = {}
        if verbose > 1:
            K = 10
            tk1 = get_topk(d1, K)
            tk2 = get_topk(d2, K)
            details['top-k'] = (tk1, tk2)
        if verbose > 2:
            details['diffs'] = self.all_diffs(d1, d2)
        if verbose > 3:
            details['all'] = (d1, d2)
        return details

    # structure of result is (result T/F, level, channel, similarity, detail)
    def compare(self, d1, d2, verbose, int8_tensor_close=True, per_axis_compare=-1):
        similarities = {}
        if d1.size != d2.size:
            return (False, self.NOT_MATCH, 0, similarities, None)

        if np.array_equal(d1, d2):
            return (True, self.EQUAL, 0, similarities, None)

        # int8 only check equal, not close
        if d1.dtype == np.int8 and int8_tensor_close:
            details = self.diff_details(d1, d2, verbose)
            return (False, self.NOT_EQUAL, 0, similarities, details)

        outer_dim = 1
        inner_dim = 1
        if per_axis_compare < len(d1.shape) and per_axis_compare >= 0:
            outer_dim = np.prod(d1.shape[0:per_axis_compare + 1])
            inner_dim = np.prod(d1.shape[per_axis_compare + 1:len(d1.shape)])
        else:
            inner_dim = np.prod(d1.shape)

        channel_simi = {}
        for loop in np.arange(outer_dim):
            if outer_dim > 1:
                d1_loop = d1.ravel()[loop * inner_dim:(loop + 1) * inner_dim]
                d2_loop = d2.ravel()[loop * inner_dim:(loop + 1) * inner_dim]
            else:
                d1_loop = d1.ravel()
                d2_loop = d2.ravel()
            simi = {}
            # check allclose
            for order in range((self.close_order_tol + 2), 1, -1):
                if (np.allclose(d1_loop, d2_loop, rtol=1 * 10**(-order), atol=1e-8,
                                equal_nan=True)):
                    break
            if order >= self.close_order_tol:
                simi["close_order"] = order
                channel_simi[loop] = (True, self.CLOSE, 0, simi, None)
                continue

            d1_loop[np.isnan(d1_loop)] = 0.0
            d2_loop[np.isnan(d2_loop)] = 0.0
            d1_loop[np.isposinf(d1_loop)] = 10000.0
            d1_loop[np.isneginf(d1_loop)] = -10000.0
            d2_loop[np.isposinf(d2_loop)] = 10000.0
            d2_loop[np.isneginf(d2_loop)] = -10000.0
            # check similarity

            def cosine_distance(a, b):
                # 计算余弦相似度
                cos_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                # 余弦距离 = 1 - 余弦相似度
                return 1 - cos_similarity

            # cosine similarity
            # cosine_similarity_my = self.cosine_similarity(d1.flatten(), d2.flatten())
            if np.sum(np.abs(d1_loop)) != 0 and np.sum(np.abs(d2_loop) != 0):
                a = d1_loop if d1_loop.dtype == np.float32 else d1_loop.astype(np.float32)
                b = d2_loop if d2_loop.dtype == np.float32 else d2_loop.astype(np.float32)
                cosine_similarity = 1 - cosine_distance(a, b)
            else:
                cosine_similarity = 0.0
            # measure euclidean similarity
            ed = self.euclidean_distance(d1_loop, d2_loop)
            sr = self.square_rooted((d1_loop + d2_loop) / 2) + 1e-7
            if (np.isinf(ed) or np.isinf(sr)):
                euclidean_similarity = 0.0
            else:
                euclidean_similarity = 1 - ed / sr

            sqnr = self.sqnr_similarity(d1_loop, d2_loop)

            simi["cosine"] = cosine_similarity
            simi["euclid"] = euclidean_similarity
            simi["sqnr"] = sqnr
            # check similarity
            if (cosine_similarity > self.cosine_similarity_tol
                    and euclidean_similarity > self.euclidean_similarity_tol
                    and sqnr > self.signal_to_quantization_noise_tol):
                channel_simi[loop] = (True, self.SIMILAR, loop, simi, None)
            else:
                # Not similar
                details = self.diff_details(d1_loop, d2_loop, verbose)
                channel_simi[loop] = (False, self.NOT_SIMILAR, loop, simi, details)

        result = channel_simi[loop]
        min_cos = 1.0
        for loop in np.arange(outer_dim):
            (r, s, c, ss, d) = channel_simi[loop]
            if (not r) and (ss['cosine'] < min_cos):
                result = channel_simi[loop]
                min_cos = ss['cosine']
        return result

    def int8_tensor_stats(self, d):
        d_int8 = d.astype(np.int8)
        pos = np.sum(d_int8 == 127)
        neg = np.sum(d_int8 == -128)
        zeros = np.sum(d_int8 == 0)
        b_low = np.sum(np.abs(d_int8) <= 8)  # 16, 32, 63
        tol = d_int8.size
        print("    pos(x=127)    = {:.4f}  [{}/{}]".format(pos / tol, pos, tol))
        print("    neg(x=-128)   = {:.4f}  [{}/{}]".format(neg / tol, neg, tol))
        print("    zeros(x=0)    = {:.4f}  [{}/{}]".format(zeros / tol, zeros, tol))
        print("    low(abs(x)<8) = {:.4f}  [{}/{}]".format(b_low / tol, b_low, tol))

    def print_result(self, d1, name, result, verbose, per_axis_compare):
        print("[{:<32}] {:>12} [{:>6}]".format(name, result[1],
                                               "PASSED" if result[0] else "FAILED"))
        if (verbose > 0):
            print("    {} {} ".format(d1.shape, d1.dtype))
            if (result[1] == self.CLOSE):
                print("    close order            = {}".format(result[3]["close_order"]))
            if (result[1] == self.SIMILAR or result[1] == self.NOT_SIMILAR):
                if per_axis_compare >= 0 and per_axis_compare < len(d1.shape):
                    print("    channel {} on axis {} :".format(result[2], per_axis_compare))
                print("    cosine_similarity      = {:.6f}".format(result[3]["cosine"]))
                print("    euclidean_similarity   = {:.6f}".format(result[3]["euclid"]))
                print("    sqnr_similarity        = {:.6f}".format(result[3]["sqnr"]))
        if d1.dtype == np.int8:
            self.int8_tensor_stats(d1)

        details = result[-1]
        if not details:
            return
        if (verbose > 1 and not result[0]):
            print('top-k:')
            print(' idx-t  target  idx-r  ref')
            tk1, tk2 = details['top-k']
            for i in range(len(tk1)):
                idx_t, target = tk1[i]
                idx_r, ref = tk2[i]
                print(" ", idx_t, target, idx_r, ref)
        if (verbose > 2 and not result[0] and details['diffs'] is not None):
            print("all-diffs:")
            print(" idx  target  ref")
            for i in details['diffs']:
                print(" ", *i)
        if (verbose > 3 and not result[0]):
            print("all-elements:")
            print(" idx  target  ref")
            target, ref = details['all']
            for index, val in np.ndenumerate(target):
                print(" ", index, val, ref[index])


class TensorCompareStats():

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = OrderedDict()
        self.count = {}
        self.count[TensorCompare.NOT_MATCH] = 0
        self.count[TensorCompare.EQUAL] = 0
        self.count[TensorCompare.NOT_EQUAL] = 0
        self.count[TensorCompare.CLOSE] = 0
        self.count[TensorCompare.SIMILAR] = 0
        self.count[TensorCompare.NOT_SIMILAR] = 0
        self.min_cosine_similarity = 1.0
        self.min_euclidean_similarity = 1.0
        self.min_sqnr = float('inf')

    def update(self, name, result):
        self.results[name] = result
        if result[0]:
            self.passed = self.passed + 1
            assert (result[1] == TensorCompare.EQUAL or result[1] == TensorCompare.CLOSE
                    or result[1] == TensorCompare.SIMILAR)
        else:
            self.failed = self.failed + 1
            assert (result[1] == TensorCompare.NOT_EQUAL or result[1] == TensorCompare.NOT_SIMILAR
                    or result[1] == TensorCompare.NOT_MATCH)
        self.count[result[1]] = self.count[result[1]] + 1
        # record min similarity
        if result[1] == TensorCompare.SIMILAR or result[1] == TensorCompare.NOT_SIMILAR:
            self.min_cosine_similarity = min(self.min_cosine_similarity, result[3]["cosine"])
            self.min_euclidean_similarity = min(self.min_euclidean_similarity, result[3]["euclid"])
            self.min_sqnr = min(self.min_sqnr, result[3]["sqnr"])

    def print_result(self):
        print("%d compared" % (len(self.results)))
        print("%d passed" % (self.passed))
        print("  %d equal, %d close, %d similar" %
              (self.count[TensorCompare.EQUAL], self.count[TensorCompare.CLOSE],
               self.count[TensorCompare.SIMILAR]))
        print("%d failed" % (self.failed))
        print("  %d not equal, %d not similar" %
              (self.count[TensorCompare.NOT_EQUAL], self.count[TensorCompare.NOT_SIMILAR]))
        print("min_similiarity = ({}, {}, {})".format(self.min_cosine_similarity,
                                                      self.min_euclidean_similarity, self.min_sqnr))

    def save_result(self, csv_file, operations, quant_types):
        has_similarity = lambda x: (x == TensorCompare.SIMILAR or x == TensorCompare.NOT_SIMILAR)
        with open(csv_file, mode='w') as f:
            f.write("name, op, quant, pass, sim_cos, sim_euc, sqnr\n")
            for name, result in self.results.items():
                op = operations.get(name, '-')
                qtype = quant_types.get(name, '-')
                is_equal = bool(result[1] == TensorCompare.EQUAL)
                is_close = bool(result[1] == TensorCompare.CLOSE)
                is_similar = bool(result[1] == TensorCompare.SIMILAR)
                is_pass = bool(is_similar or is_close or is_equal)
                cos = float(result[3]["cosine"]) if has_similarity(result[1]) else 1.0
                euc = float(result[3]["euclid"]) if has_similarity(result[1]) else 1.0
                sqnr = float(result[3]["sqnr"]) if has_similarity(result[1]) else float('inf')
                f.write("{}, {}, {}, {}, {}, {}, {}\n".format(name, op, qtype, is_pass, cos, euc,
                                                              sqnr))
