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

void py_cuda::cudaInstanceNormOp(top::InstanceNormOp op) {
  auto in_shape = module::getShape(op.getInput());
  int channel = in_shape[1];
  float eps = op.getEps().convertToDouble();

  int outer_dim = 1;
  for (int i = 0; i < 2; i++) outer_dim *= in_shape[i];
  int inner_dim = 1;
  for (int i = 2; i < (int)in_shape.size(); i++) inner_dim *= in_shape[i];

  void *w_ptr = module::isNone(op.getWeight()) ? nullptr
                : getCudaData(op.getWeight());
  void *b_ptr = module::isNone(op.getBias()) ? nullptr
                : getCudaData(op.getBias());

  cuda::bmInstanceNorm(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                        w_ptr, b_ptr, outer_dim, inner_dim, channel, eps);
}

void py_cuda::cudaInstanceNormOp(tpu::InstanceNormOp op) {
  auto in_shape = module::getShape(op.getInput());
  int channel = in_shape[1];
  float eps = op.getEps().convertToDouble();

  int outer_dim = 1;
  for (int i = 0; i < 2; i++) outer_dim *= in_shape[i];
  int inner_dim = 1;
  for (int i = 2; i < (int)in_shape.size(); i++) inner_dim *= in_shape[i];

  void *w_ptr = module::isNone(op.getWeight()) ? nullptr
                : getCudaData(op.getWeight());
  void *b_ptr = module::isNone(op.getBias()) ? nullptr
                : getCudaData(op.getBias());

  cuda::bmInstanceNorm(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                        w_ptr, b_ptr, outer_dim, inner_dim, channel, eps);
}
