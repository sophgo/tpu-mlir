#!/usr/bin/env python3

import numpy as np
import sys
import struct
# from math import fabs
from enum import IntEnum
from scipy import spatial
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
  NOT_MATCH   = "NOT_MATCH"
  EQUAL       = "EQUAL"
  NOT_EQUAL   = "NOT_EQUAL"
  CLOSE       = "CLOSE"
  SIMILAR     = "SIMILAR"
  NOT_SIMILAR = "NOT_SIMLIAR"

  def __init__(self, close_order_tol=3,
               cosine_similarity_tol = 0.99,
               correlation_similarity_tol = 0.99,
               euclidean_similarity_tol = 0.90,
               signal_to_quantization_noise_tol = 50):
    self.close_order_tol            = close_order_tol
    self.cosine_similarity_tol      = cosine_similarity_tol
    self.correlation_similarity_tol = correlation_similarity_tol
    self.euclidean_similarity_tol   = euclidean_similarity_tol
    self.signal_to_quantization_noise_tol   = signal_to_quantization_noise_tol
    return

  def square_rooted(self, x):
    return sqrt(sum([a*a for a in x]))

  def cosine_similarity(self, x, y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = self.square_rooted(x)*self.square_rooted(y)
    return round(numerator/float(denominator),3)

  def euclidean_distance(self, x, y):
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

  def sqnr_similarity(self, signal_raw, signal_dequant, remove_zero=True):
    # SQNR is non-commutative
    # Unlike other distance function
    # Cannot change the order of signal_raw and signal_dequant
    raw = signal_raw.flatten()
    dequant = signal_dequant.flatten()

    if remove_zero is True:
        idx = raw != 0
        raw = raw[idx]
        dequant = dequant[idx]

    noise = raw - dequant

    avg_raw = np.sum(raw) / raw.size
    avg_noise = np.sum(noise) / noise.size

    raw_zero_mean = raw - avg_raw
    noise_zero_mean = noise - avg_noise

    var_raw_zero_mean = np.sum(np.square(raw_zero_mean))
    var_noise_zero_mean = np.sum(np.square(noise_zero_mean))
    if var_noise_zero_mean == 0 or var_raw_zero_mean == 0:
      return float('inf')
    sqnr = 10 * np.log10(var_raw_zero_mean / var_noise_zero_mean)

    return sqnr

  def all_diffs(self, d1, d2):
    diffs = list()
    d1f = d1.flatten()
    d2f = d2.flatten()
    if d1f.dtype == np.int8:
      assert(d2f.dtype == np.int8)
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
      details['diffs'] = self.all_diffs(d1,d2)
    if verbose > 3:
      details['all'] = (d1, d2)
    return details

  def compare(self, d1, d2, verbose, int8_tensor_close=True):
    similarities = {}
    if d1.size != d2.size:
      return (False, self.NOT_MATCH, similarities, None)

    if np.array_equal(d1, d2):
      return (True, self.EQUAL, similarities, None)

    # int8 only check equal, not close
    if d1.dtype == np.int8 and int8_tensor_close:
      details = self.diff_details(d1, d2, verbose)
      return (False, self.NOT_EQUAL, similarities, details)

    # check allclose
    for order in range((self.close_order_tol + 2), 1, -1):
      if (np.allclose(d1, d2, rtol=1 * 10**(-order), atol=1e-8, equal_nan=True)):
        break
    if order >= self.close_order_tol:
      similarities["close_order"] = order
      return (True, self.CLOSE, similarities, None)

    # check similarity
    # cosine similarity
    # cosine_similarity_my = self.cosine_similarity(d1.flatten(), d2.flatten())
    cosine_similarity = 1 - spatial.distance.cosine(d1.flatten().astype(np.float32),
                                                    d2.flatten().astype(np.float32))
    # correlation similarity
    #1 - spatial.distance.correlation(d1.flatten(), d2.flatten())
    correlation_similarity = cosine_similarity
    # measure euclidean similarity
    m = (d1+d2)/2
    ed = self.euclidean_distance(d1.flatten(), d2.flatten())
    sr = self.square_rooted(m.flatten())
    euclidean_similarity = 1 - ed / sr

    sqnr = self.sqnr_similarity(d1, d2)

    similarities["cosine"] = cosine_similarity
    similarities["correlation"] = correlation_similarity
    similarities["euclid"] = euclidean_similarity
    similarities["sqnr"] = sqnr
    # check similarity
    if (cosine_similarity > self.cosine_similarity_tol
        and correlation_similarity > self.correlation_similarity_tol
        and euclidean_similarity > self.euclidean_similarity_tol
        and sqnr > self.signal_to_quantization_noise_tol):
      return (True, self.SIMILAR, similarities, None)
    else:
      # Not similar
      details = self.diff_details(d1, d2, verbose)
      return (False, self.NOT_SIMILAR, similarities, details)

  def int8_tensor_stats(self, d):
    d_int8 = d.astype(np.int8)
    pos = np.sum(d_int8 == 127)
    neg = np.sum(d_int8 == -128)
    zeros = np.sum(d_int8 == 0)
    b_low = np.sum(np.abs(d_int8) <= 8) # 16, 32, 63
    tol = d_int8.size
    print("    pos(x=127)    = {:.4f}  [{}/{}]".format(pos / tol, pos, tol))
    print("    neg(x=-128)   = {:.4f}  [{}/{}]".format(neg / tol, neg, tol))
    print("    zeros(x=0)    = {:.4f}  [{}/{}]".format(zeros / tol, zeros, tol))
    print("    low(abs(x)<8) = {:.4f}  [{}/{}]".format(b_low / tol, b_low, tol))

  def print_result(self, d1, name, result, verbose):
    print("[{:<32}] {:>12} [{:>6}]".format(name, result[1],
           "PASSED" if result[0] else "FAILED"))
    if (verbose > 0):
      print("    {} {} ".format(d1.shape, d1.dtype))
      if (result[1] == self.CLOSE):
        print("    close order            = {}".format(result[2]["close_order"]))
      if (result[1] == self.SIMILAR or result[1] == self.NOT_SIMILAR):
        print("    cosine_similarity      = {:.6f}".format(result[2]["cosine"]))
        print("    correlation_similarity = {:.6f}".format(result[2]["correlation"]))
        print("    euclidean_similarity   = {:.6f}".format(result[2]["euclid"]))
        print("    sqnr_similarity        = {:.6f}".format(result[2]["sqnr"]))
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
    self.min_correlation_similarity = 1.0
    self.min_euclidean_similarity = 1.0
    self.min_sqnr = float('inf')

  def update(self, name, result):
    self.results[name] = result
    if result[0]:
      self.passed = self.passed + 1
      assert (result[1] == TensorCompare.EQUAL
              or result[1] == TensorCompare.CLOSE
              or result[1] == TensorCompare.SIMILAR)
    else:
      self.failed = self.failed + 1
      assert (result[1] == TensorCompare.NOT_EQUAL
              or result[1] == TensorCompare.NOT_SIMILAR)
    self.count[result[1]] = self.count[result[1]] + 1
    # record min similarity
    if result[1] == TensorCompare.SIMILAR or result[1] == TensorCompare.NOT_SIMILAR:
      self.min_cosine_similarity = min(self.min_cosine_similarity, result[2]["cosine"])
      self.min_correlation_similarity = min(self.min_correlation_similarity, result[2]["correlation"])
      self.min_euclidean_similarity = min(self.min_euclidean_similarity, result[2]["euclid"])
      self.min_sqnr = min(self.min_sqnr, result[2]["sqnr"])

  def print_result(self):
    print("%d compared"%(len(self.results)))
    print("%d passed"%(self.passed))
    print("  %d equal, %d close, %d similar"
          %(self.count[TensorCompare.EQUAL],
            self.count[TensorCompare.CLOSE],
            self.count[TensorCompare.SIMILAR]))
    print("%d failed"%(self.failed))
    print("  %d not equal, %d not similar"
          %(self.count[TensorCompare.NOT_EQUAL],
            self.count[TensorCompare.NOT_SIMILAR]))
    print("min_similiarity = ({}, {}, {}, {})".format(
            self.min_cosine_similarity,
            self.min_correlation_similarity,
            self.min_euclidean_similarity,
            self.min_sqnr))

  def save_result(self, csv_file, operations, quant_types):
    has_similarity = lambda x: (x == TensorCompare.SIMILAR
                                or x == TensorCompare.NOT_SIMILAR)
    with open(csv_file, mode='w') as f:
      f.write("name, op, quant, pass, sim_cos, sim_euc, sqnr\n")
      for name, result in self.results.items():
        op = operations.get(name, '-')
        qtype = quant_types.get(name, '-')
        is_equal = bool(result[1] == TensorCompare.EQUAL)
        is_close = bool(result[1] == TensorCompare.CLOSE)
        is_similar = bool(result[1] == TensorCompare.SIMILAR)
        is_pass = bool(is_similar or is_close or is_equal)
        cos = float(result[2]["cosine"]) if has_similarity(result[1]) else 1.0
        euc = float(result[2]["euclid"]) if has_similarity(result[1]) else 1.0
        sqnr = float(result[2]["sqnr"]) if has_similarity(result[1]) else float('-inf')
        f.write("{}, {}, {}, {}, {}, {}, {}\n".format(
          name, op, qtype, is_pass, cos, euc, sqnr))
