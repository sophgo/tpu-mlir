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

void py_cuda::cudaMeanRstdOp(top::MeanRstdOp op) {
  auto input = getCudaData(op.getInput());
  auto running_mean = getCudaData(op.getRunningMean());
  auto running_var = getCudaData(op.getRunningVar());
  auto weight = getCudaData(op.getWeight());
  auto bias = getCudaData(op.getBias());
  auto mean_out = getCudaData(op.getMean());
  auto rstd_out = getCudaData(op.getRstd());
  auto rm_update = getCudaData(op.getRunningMeanUpdate());
  auto rv_update = getCudaData(op.getRunningVarUpdate());
  auto scale_out = getCudaData(op.getScale());
  auto bias_out = getCudaData(op.getBiasNew());

  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);
  float eps = op.getEps().convertToDouble();
  float momentum = op.getMomentum().convertToDouble();

  // Stage 1: compute per-channel mean and rstd + update running stats
  cuda::meanRstd(input, mean_out, rstd_out, running_mean, running_var,
                 weight, bias, n, c, h * w, eps, momentum);

  // Stage 2: copy weight→scale, bias→bias_new, running_mean→rm_update, running_var→rv_update
  auto c_bytes = c * sizeof(float);
  CHECK_CUDA(cudaMemcpy(scale_out, weight, c_bytes, cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(bias_out, bias, c_bytes, cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(rm_update, running_mean, c_bytes, cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(rv_update, running_var, c_bytes, cudaMemcpyDeviceToDevice));
}
