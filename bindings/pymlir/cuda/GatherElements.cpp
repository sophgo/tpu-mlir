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

void py_cuda::cudaGatherElementsOp(top::GatherElementsOp op) {
  auto in_shape = module::getShape(op.getInput());
  auto idx_shape = module::getShape(op.getIndices());
  int rank = in_shape.size();
  int axis = op.getAxis();
  if (axis < 0) axis += rank;

  std::vector<int> in_strides(rank), out_strides(rank), out_shape(rank);
  int is = 1, os = 1;
  for (int d = rank - 1; d >= 0; d--) {
    in_strides[d] = is; is *= in_shape[d];
    out_strides[d] = os; os *= idx_shape[d];
    out_shape[d] = idx_shape[d];
  }

  auto idx_f32 = newCudaData(op.getIndices(), cuda::DT_F32);
  cuda::bmGatherElements(getCudaData(op.getInput()), idx_f32.get(),
                          getCudaData(op.getOutput()),
                          out_shape.data(), in_strides.data(), out_strides.data(),
                          rank, axis);
}

void py_cuda::cudaGatherElementsOp(tpu::GatherElementsOp op) {
  auto in_shape = module::getShape(op.getInput());
  auto idx_shape = module::getShape(op.getIndices());
  int rank = in_shape.size();
  int axis = op.getAxis();
  if (axis < 0) axis += rank;

  std::vector<int> in_strides(rank), out_strides(rank), out_shape(rank);
  int is = 1, os = 1;
  for (int d = rank - 1; d >= 0; d--) {
    in_strides[d] = is; is *= in_shape[d];
    out_strides[d] = os; os *= idx_shape[d];
    out_shape[d] = idx_shape[d];
  }

  auto idx_f32 = newCudaData(op.getIndices(), cuda::DT_F32);
  cuda::bmGatherElements(getCudaData(op.getInput()), idx_f32.get(),
                          getCudaData(op.getOutput()),
                          out_shape.data(), in_strides.data(), out_strides.data(),
                          rank, axis);
}
