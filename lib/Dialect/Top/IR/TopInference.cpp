//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/Dnnl/Conv.h"
#include "sophgo/Support/Dnnl/Pool.h"
#include "sophgo/Support/Dnnl/MatMul.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "dnnl.hpp"
#include "omp.h"

using namespace sophgo;
using namespace mlir;

int omp_schedule(int count) {
  return (count + omp_get_num_threads() - 1) / omp_get_num_threads();
}

template <typename T> static void relu(T *src, T *dst, size_t size) {
#pragma omp parallel for schedule(static, omp_schedule(size))
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
  }
}

LogicalResult top::ConvOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], n, ic, ih,
              iw, oc, oh, ow, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, g);
  p.handle = (void *)conv;
  return success();
}

void top::ConvOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto conv = (Conv *)p.handle;
    delete conv;
    p.handle = nullptr;
  }
}

LogicalResult top::ConvOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  conv->run();
  if (do_relu()) {
    size_t num_elem =
        output().getType().cast<RankedTensorType>().getNumElements();
    relu(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}

LogicalResult top::ReluOp::inference(InferenceParameter &p) {
  auto num_elem = input().getType().cast<RankedTensorType>().getNumElements();
  relu(p.inputs[0], p.outputs[0], num_elem);
  return success();
}

LogicalResult top::AddOp::inference(InferenceParameter &p) {
  auto num_elem = output().getType().cast<RankedTensorType>().getNumElements();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = 0;
    for (auto in : p.inputs) {
      if (in != nullptr) {
        p.outputs[0][i] += in[i];
      }
    }
  }
  if (do_relu()) {
    relu(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}

LogicalResult top::MaxPoolOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             is_global, count_include_pad);
  pooling->setup(p.inputs[0], p.outputs[0], n, c, ih, iw, oh, ow, kh, kw, sh,
                 sw, pt, pb, pl, pr, false, count_include_pad, pad_value);
  p.handle = (void *)pooling;
  return success();
}

void top::MaxPoolOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult top::MaxPoolOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto pooling = (Pooling *)p.handle;
  pooling->run();
  if (do_relu()) {
    size_t num_elem =
        output().getType().cast<RankedTensorType>().getNumElements();
    relu(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}

LogicalResult top::AvgPoolOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             is_global, count_include_pad);
  pooling->setup(p.inputs[0], p.outputs[0], n, c, ih, iw, oh, ow, kh, kw, sh,
                 sw, pt, pb, pl, pr, true, count_include_pad, pad_value);
  p.handle = (void *)pooling;
  return success();
}

void top::AvgPoolOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult top::AvgPoolOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto pooling = (Pooling *)p.handle;
  pooling->run();
  if (do_relu()) {
    size_t num_elem =
        output().getType().cast<RankedTensorType>().getNumElements();
    relu(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}

LogicalResult top::ReshapeOp::inference(InferenceParameter &p) {
  auto num_elem = output().getType().cast<RankedTensorType>().getNumElements();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = p.inputs[0][i];
  }
  return success();
}

LogicalResult top::MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  int64_t batch, M, K, N;
  parseParam(batch, M, K, N);
  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], batch, M,
                K, N, do_relu());
  p.handle = (void *)matmul;
  return success();
}

void top::MatMulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto matmul = (MatMul *)p.handle;
    delete matmul;
    p.handle = nullptr;
  }
  return;
}

LogicalResult top::MatMulOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto matmul = (MatMul *)p.handle;
  matmul->run();
  return success();
}
