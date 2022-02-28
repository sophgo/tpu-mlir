//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Tops/IR/TopsOps.h"
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/DnnlConv.h"
#include "sophgo/Support/DnnlPool.h"
#include "sophgo/Support/DnnlMatMul.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "dnnl.hpp"
#include "omp.h"
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

LogicalResult tops::ConvOp::init() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  auto conv = new dnnl::Conv();
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  conv->setup(infer_op.input_buffers[0], infer_op.input_buffers[1],
              infer_op.input_buffers[2], infer_op.output_buffers[0], n, ic, ih,
              iw, oc, oh, ow, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, g);
  infer_op.handle = (void *)conv;
  return success();
}

void tops::ConvOp::deinit() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  if (infer_op.handle != nullptr) {
    auto conv = (dnnl::Conv *)infer_op.handle;
    delete conv;
    infer_op.handle = nullptr;
  }
}

LogicalResult tops::ConvOp::inference() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  if (infer_op.handle == nullptr) {
    return failure();
  }
  auto conv = (dnnl::Conv *)infer_op.handle;
  conv->run();
  if (do_relu()) {
    size_t num_elem =
        output().getType().cast<RankedTensorType>().getNumElements();
    relu(infer_op.output_buffers[0], infer_op.output_buffers[0], num_elem);
  }
  return success();
}

LogicalResult tops::ReluOp::inference() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  auto num_elem = input().getType().cast<RankedTensorType>().getNumElements();
  float *input = infer_op.input_buffers[0];
  float *output = infer_op.output_buffers[0];
  relu(input, output, num_elem);
  return success();
}

LogicalResult tops::AddOp::inference() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  auto num_elem = output().getType().cast<RankedTensorType>().getNumElements();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    infer_op.output_buffers[0][i] = 0;
    for (auto in : infer_op.input_buffers) {
      if (in != nullptr) {
        infer_op.output_buffers[0][i] += in[i];
      }
    }
  }
  return success();
}

LogicalResult tops::MaxPoolOp::init() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  auto pooling = new dnnl::Pooling();
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             is_global, count_include_pad);
  pooling->setup(infer_op.input_buffers[0], infer_op.output_buffers[0], n, c,
                 ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, false,
                 count_include_pad, pad_value);
  infer_op.handle = (void *)pooling;
  return success();
}

void tops::MaxPoolOp::deinit() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  if (infer_op.handle != nullptr) {
    auto pooling = (dnnl::Pooling *)infer_op.handle;
    delete pooling;
    infer_op.handle = nullptr;
  }
  return;
}

LogicalResult tops::MaxPoolOp::inference() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  if (infer_op.handle == nullptr) {
    return failure();
  }
  auto pooling = (dnnl::Pooling *)infer_op.handle;
  pooling->run();
  if (do_relu()) {
    size_t num_elem =
        output().getType().cast<RankedTensorType>().getNumElements();
    relu(infer_op.output_buffers[0], infer_op.output_buffers[0], num_elem);
  }
  return success();
}

LogicalResult tops::AvgPoolOp::init() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  auto pooling = new dnnl::Pooling();
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             is_global, count_include_pad);
  pooling->setup(infer_op.input_buffers[0], infer_op.output_buffers[0], n, c,
                 ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, true,
                 count_include_pad, pad_value);
  infer_op.handle = (void *)pooling;
  return success();
}

void tops::AvgPoolOp::deinit() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  if (infer_op.handle != nullptr) {
    auto pooling = (dnnl::Pooling *)infer_op.handle;
    delete pooling;
    infer_op.handle = nullptr;
  }
  return;
}

LogicalResult tops::AvgPoolOp::inference() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  if (infer_op.handle == nullptr) {
    return failure();
  }
  auto pooling = (dnnl::Pooling *)infer_op.handle;
  pooling->run();
  if (do_relu()) {
    size_t num_elem =
        output().getType().cast<RankedTensorType>().getNumElements();
    relu(infer_op.output_buffers[0], infer_op.output_buffers[0], num_elem);
  }
  return success();
}

LogicalResult tops::ReshapeOp::inference() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  auto num_elem = output().getType().cast<RankedTensorType>().getNumElements();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    infer_op.output_buffers[0][i] = infer_op.input_buffers[0][i];
  }
  return success();
}

LogicalResult tops::MatMulOp::init() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  auto matmul = new dnnl::MatMul();
  int64_t batch, M, K, N;
  parseParam(batch, M, K, N);
  matmul->setup(infer_op.input_buffers[0], infer_op.input_buffers[1],
                infer_op.input_buffers[2], infer_op.output_buffers[0], batch, M,
                K, N, do_relu());
  infer_op.handle = (void *)matmul;
  return success();
}

void tops::MatMulOp::deinit() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  if (infer_op.handle != nullptr) {
    auto matmul = (dnnl::MatMul *)infer_op.handle;
    delete matmul;
    infer_op.handle = nullptr;
  }
  return;
}

LogicalResult tops::MatMulOp::inference() {
  auto infer_op = llvm::cast<mlir::InferenceInterface>(this->getOperation());
  if (infer_op.handle == nullptr) {
    return failure();
  }
  auto matmul = (dnnl::MatMul *)infer_op.handle;
  matmul->run();
  return success();
}
