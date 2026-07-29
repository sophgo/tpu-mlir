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
#include <vector>

void py_cuda::cudaMeanStdScaleOp(top::MeanStdScaleOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);

  // Read per-channel host-side attributes, upload to device
  auto mean_attr = module::getF64Array(op.getMean());
  auto std_attr = module::getF64Array(op.getStd());
  auto scale_attr = module::getF64Array(op.getScale());
  auto zp_attr = module::getF64Array(op.getZeroPoints());

  std::vector<float> mean_c(c), std_c(c), scale_c(c), zp_c(c);
  for (int i = 0; i < c; i++) {
    mean_c[i] = mean_attr->at(i % mean_attr->size());
    std_c[i]  = std_attr->at(i % std_attr->size());
    scale_c[i] = scale_attr->at(i % scale_attr->size());
    zp_c[i]   = zp_attr->at(i % zp_attr->size());
  }
  auto mean_d = cuda_malloc(c * sizeof(float));
  auto std_d  = cuda_malloc(c * sizeof(float));
  auto scale_d = cuda_malloc(c * sizeof(float));
  auto zp_d   = cuda_malloc(c * sizeof(float));
  CHECK_CUDA(cudaMemcpy(mean_d.get(), mean_c.data(), c * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(std_d.get(),  std_c.data(),  c * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(scale_d.get(), scale_c.data(), c * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(zp_d.get(),   zp_c.data(),   c * sizeof(float), cudaMemcpyHostToDevice));

  cuda::meanStdScale(input, output, mean_d.get(), std_d.get(), scale_d.get(), zp_d.get(), n, c, h, w);
}
