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

void py_cuda::cudaReverseOp(top::ReverseOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto shape = module::getShape(op.getOutput());
  int64_t axis = op.getAxis();
  if (axis < 0)
    axis += shape.size();

  int outer_stride = 1;
  for (int i = 0; i < axis; ++i)
    outer_stride *= shape[i];
  int axis_dim = shape[axis];
  int inner_stride = 1;
  for (int i = axis + 1; i < (int)shape.size(); ++i)
    inner_stride *= shape[i];

  cuda::bmReverse(input, output, outer_stride, axis_dim, inner_stride);
}

void py_cuda::cudaReverseOp(tpu::ReverseOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto shape = module::getShape(op.getOutput());
  int64_t axis = op.getAxis();
  if (axis < 0)
    axis += shape.size();

  int outer_stride = 1;
  for (int i = 0; i < axis; ++i)
    outer_stride *= shape[i];
  int axis_dim = shape[axis];
  int inner_stride = 1;
  for (int i = axis + 1; i < (int)shape.size(); ++i)
    inner_stride *= shape[i];

  cuda::bmReverse(input, output, outer_stride, axis_dim, inner_stride);
}
