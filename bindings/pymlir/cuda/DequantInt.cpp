//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

using tpu_mlir::cuda::rounding_mode_t;
using tpu_mlir::cuda::RD_HALF_AWAY_FROM_ZERO;

static int getQuantModeInt(const std::string &s) {
  return (s == "Normal") ? 0 : 1;
}

static rounding_mode_t getRoundMode(const std::string &s) {
  if (s == "HalfUp")           return tpu_mlir::cuda::RD_HALF_UP;
  if (s == "HalfDown")         return tpu_mlir::cuda::RD_HALF_DOWN;
  if (s == "HalfToEven")       return tpu_mlir::cuda::RD_HALF_TO_EVEN;
  if (s == "HalfToOdd")        return tpu_mlir::cuda::RD_HALF_TO_ODD;
  if (s == "HalfTowardsZero")  return tpu_mlir::cuda::RD_HALF_TOWARDS_ZERO;
  if (s == "TowardsZero")      return tpu_mlir::cuda::RD_TOWARDS_ZERO;
  if (s == "Up")               return tpu_mlir::cuda::RD_UP;
  if (s == "Down")             return tpu_mlir::cuda::RD_DOWN;
  return RD_HALF_AWAY_FROM_ZERO;
}

// ==========================================================================
// tpu::DequantIntOp — multiplier/shift are scalars (SI32Attr / I64Attr)
// ==========================================================================

void py_cuda::cudaDequantIntOp(tpu::DequantIntOp op) {
  auto in_type = getCudaType(op.getInput());
  auto qtype = module::getUniformQuantizedType(op.getInput());
  int32_t zp = qtype.getZeroPoint();
  int num = module::getNumElements(op.getInput());

  int64_t multiplier = op.getMultiplier();
  int64_t shift = op.getShift();
  int64_t lshift = op.getLshift();

  int mode = (op.getQuantMode() == tpu::DequantMode::Normal) ? 0 : 1;
  rounding_mode_t rmode = RD_HALF_AWAY_FROM_ZERO;
  auto tpuRound = op.getRoundMode();
  if (tpuRound == tpu::RoundMode::HalfUp) rmode = tpu_mlir::cuda::RD_HALF_UP;
  else if (tpuRound == tpu::RoundMode::HalfDown) rmode = tpu_mlir::cuda::RD_HALF_DOWN;
  else if (tpuRound == tpu::RoundMode::HalfToEven) rmode = tpu_mlir::cuda::RD_HALF_TO_EVEN;
  else if (tpuRound == tpu::RoundMode::HalfToOdd) rmode = tpu_mlir::cuda::RD_HALF_TO_ODD;
  else if (tpuRound == tpu::RoundMode::HalfTowardsZero) rmode = tpu_mlir::cuda::RD_HALF_TOWARDS_ZERO;
  else if (tpuRound == tpu::RoundMode::TowardsZero) rmode = tpu_mlir::cuda::RD_TOWARDS_ZERO;
  else if (tpuRound == tpu::RoundMode::Up) rmode = tpu_mlir::cuda::RD_UP;
  else if (tpuRound == tpu::RoundMode::Down) rmode = tpu_mlir::cuda::RD_DOWN;

  cuda::bmDequantIntPerTensor(getCudaData(op.getInput()),
                               getCudaData(op.getOutput()),
                               num, multiplier, shift, lshift,
                               zp, mode, rmode, in_type);
}

// ==========================================================================
// top::DequantIntOp — multiplier/shift are arrays (I64ArrayAttr)
// ==========================================================================

void py_cuda::cudaDequantIntOp(top::DequantIntOp op) {
  auto in_type = getCudaType(op.getInput());
  auto qtype = module::getUniformQuantizedType(op.getInput());
  int32_t zp = qtype.getZeroPoint();
  int num = module::getNumElements(op.getInput());
  auto shape = module::getShape(op.getInput());

  auto multi = module::getI64Array(op.getMultiplier());
  auto shift = module::getI64Array(op.getShift());
  int64_t lshift = op.getLshift();

  int mode = getQuantModeInt(op.getQuantModeAttr().str());
  rounding_mode_t rmode = getRoundMode(op.getRoundModeAttr().str());

  bool is_per_channel = multi->size() > 1;
  if (!is_per_channel) {
    cuda::bmDequantIntPerTensor(getCudaData(op.getInput()),
                                 getCudaData(op.getOutput()),
                                 num, multi->at(0), shift->at(0), lshift,
                                 zp, mode, rmode, in_type);
  } else {
    int channel_dim = shape[1];
    int outer_dim = shape[0];
    int inner_dim = 1;
    for (int i = 2; i < (int)shape.size(); ++i) inner_dim *= shape[i];

    auto d_multi = cuda_malloc(channel_dim * sizeof(int64_t));
    auto d_shift = cuda_malloc(channel_dim * sizeof(int64_t));
    CHECK_CUDA(cudaMemcpy(d_multi.get(), multi->data(),
                          channel_dim * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_shift.get(), shift->data(),
                          channel_dim * sizeof(int64_t),
                          cudaMemcpyHostToDevice));

    cuda::bmDequantIntPerChannel(
        getCudaData(op.getInput()), getCudaData(op.getOutput()),
        outer_dim, channel_dim, inner_dim,
        (int64_t *)d_multi.get(), (int64_t *)d_shift.get(),
        lshift, zp, mode, rmode, in_type);
  }
}
