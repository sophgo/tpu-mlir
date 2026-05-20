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

void py_cuda::cudaArgOp(tpu::ArgOp op) {
  auto input = op.getInput();
  auto indices = op.getIndices();
  auto in_shape = module::getShape(input);
  int axis = op.getAxis();
  int ndim = in_shape.size();
  if (axis < 0) axis += ndim;

  int outer_dim = 1;
  for (int i = 0; i < axis; i++) outer_dim *= in_shape[i];
  int axis_dim = in_shape[axis];
  int inner_dim = 1;
  for (int i = axis + 1; i < ndim; i++) inner_dim *= in_shape[i];

  if (!module::getStorageType(input).isF32()) {
    UNREACHABLE_OP("Not Implemented", op);
  }

  bool select_last = op.getSelectLastIndex();
  auto mode = op.getMode().str();
  if (mode == "ArgMax") {
    cuda::bmArgMax(getCudaData(input), getCudaData(indices),
                   outer_dim, axis_dim, inner_dim, select_last);
  } else {
    cuda::bmArgMin(getCudaData(input), getCudaData(indices),
                   outer_dim, axis_dim, inner_dim, select_last);
  }
}

void py_cuda::cudaArgOp(top::ArgOp op) {
  auto input = op.getInput();
  auto indices = op.getIndices();
  auto in_shape = module::getShape(input);
  int axis = op.getAxis();
  int ndim = in_shape.size();
  if (axis < 0) axis += ndim;

  int outer_dim = 1;
  for (int i = 0; i < axis; i++) outer_dim *= in_shape[i];
  int axis_dim = in_shape[axis];
  int inner_dim = 1;
  for (int i = axis + 1; i < ndim; i++) inner_dim *= in_shape[i];

  if (!module::getStorageType(input).isF32()) {
    UNREACHABLE_OP("Not Implemented", op);
  }

  bool select_last = op.getSelectLastIndex();
  auto mode = op.getMode().str();
  if (mode == "ArgMax") {
    cuda::bmArgMax(getCudaData(input), getCudaData(indices),
                   outer_dim, axis_dim, inner_dim, select_last);
  } else {
    cuda::bmArgMin(getCudaData(input), getCudaData(indices),
                   outer_dim, axis_dim, inner_dim, select_last);
  }
}
