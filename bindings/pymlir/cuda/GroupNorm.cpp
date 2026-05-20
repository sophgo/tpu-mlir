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

void py_cuda::cudaGroupNormOp(top::GroupNormOp op) {
  auto in_shape = module::getShape(op.getInput());
  int channel = in_shape[1];
  int num_groups = op.getNumGroups();
  int channel_per_group = channel / num_groups;
  float eps = op.getEps().convertToDouble();

  int outer_dim = in_shape[0] * num_groups;
  int inner_dim = channel_per_group;
  for (int i = 2; i < (int)in_shape.size(); i++) inner_dim *= in_shape[i];

  void *w_ptr = module::isNone(op.getWeight()) ? nullptr
                : getCudaData(op.getWeight());
  void *b_ptr = module::isNone(op.getBias()) ? nullptr
                : getCudaData(op.getBias());

  cuda::bmGroupNorm(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                     w_ptr, b_ptr,
                     outer_dim, inner_dim, channel, channel_per_group, eps);
}

void py_cuda::cudaGroupNormOp(tpu::GroupNormOp op) {
  auto in_shape = module::getShape(op.getInput());
  int channel = in_shape[1];
  int num_groups = op.getNumGroups();
  int channel_per_group = channel / num_groups;
  float eps = op.getEps().convertToDouble();

  int outer_dim = in_shape[0] * num_groups;
  int inner_dim = channel_per_group;
  for (int i = 2; i < (int)in_shape.size(); i++) inner_dim *= in_shape[i];

  void *w_ptr = module::isNone(op.getWeight()) ? nullptr
                : getCudaData(op.getWeight());
  void *b_ptr = module::isNone(op.getBias()) ? nullptr
                : getCudaData(op.getBias());

  cuda::bmGroupNorm(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                     w_ptr, b_ptr,
                     outer_dim, inner_dim, channel, channel_per_group, eps);
}
