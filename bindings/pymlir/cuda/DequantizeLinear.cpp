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

void py_cuda::cudaDequantizeLinearOp(top::DequantizeLinearOp op) {
  auto in_type = getCudaType(op.getInput());
  int num = module::getNumElements(op.getInput());
  auto scale = module::getF64Array(op.getXScale());
  auto zp = module::getI32Array(op.getXZeroPoint());
  auto shape = module::getShape(op.getInput());

  bool is_per_channel = scale->size() > 1;
  if (!is_per_channel) {
    cuda::bmDequantizeLinearPerTensor(getCudaData(op.getInput()),
                                       getCudaData(op.getOutput()),
                                       (float)scale->at(0), zp->at(0),
                                       num, in_type);
  } else {
    int axis = op.getAxis();
    if (axis < 0) axis += shape.size();
    int outer_dim = 1, channel_dim = shape[axis], inner_dim = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= shape[i];
    for (int i = axis + 1; i < (int)shape.size(); ++i) inner_dim *= shape[i];

    // copy scale/zp to GPU
    auto d_scale = cuda_malloc(channel_dim * sizeof(float));
    auto d_zp = cuda_malloc(channel_dim * sizeof(int32_t));
    std::vector<float> s(channel_dim);
    std::vector<int32_t> z(channel_dim);
    for (int i = 0; i < channel_dim; ++i) {
      s[i] = (float)scale->at(i);
      z[i] = zp->at(i);
    }
    CHECK_CUDA(cudaMemcpy(d_scale.get(), s.data(),
                          channel_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_zp.get(), z.data(),
                          channel_dim * sizeof(int32_t), cudaMemcpyHostToDevice));

    cuda::bmDequantizeLinearPerChannel(
        getCudaData(op.getInput()), getCudaData(op.getOutput()),
        (float *)d_scale.get(), (int32_t *)d_zp.get(),
        outer_dim, channel_dim, inner_dim, in_type);
  }
}
