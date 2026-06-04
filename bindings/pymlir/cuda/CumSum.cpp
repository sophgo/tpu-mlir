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

static void paramsFromShape(const std::vector<int64_t> &shape, int axis,
                            int &outer_dim, int &axis_dim, int &stride) {
  int num_dims = shape.size();
  if (axis < 0) axis += num_dims;
  outer_dim = 1;
  for (int i = 0; i < axis; ++i) outer_dim *= shape[i];
  axis_dim = shape[axis];
  stride = 1;
  for (int i = axis + 1; i < num_dims; ++i) stride *= shape[i];
}

void py_cuda::cudaCumSumOp(top::CumSumOp op) {
  auto shape = module::getShape(op.getInput());
  int outer_dim, axis_dim, stride;
  paramsFromShape(shape, op.getAxis(), outer_dim, axis_dim, stride);
  cuda::bmCumSum(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                  outer_dim, axis_dim, stride);
}

void py_cuda::cudaCumSumOp(tpu::CumSumOp op) {
  auto shape = module::getShape(op.getInput());
  int outer_dim, axis_dim, stride;
  paramsFromShape(shape, op.getAxis(), outer_dim, axis_dim, stride);
  cuda::bmCumSum(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                  outer_dim, axis_dim, stride);
}
