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

void py_cuda::cudaIndexPutOp(top::IndexPutOp op) {
  auto in_shape = module::getShape(op.getInput());
  int num_indices = module::getNumElements(op.getIndices());
  int inner_dim = 1;
  for (int i = 1; i < (int)in_shape.size(); i++) inner_dim *= in_shape[i];
  bool accumulate = op.getAccumulate();

  // copy input to output first
  size_t bytes = module::getNumElements(op.getOutput()) * sizeof(float);
  cudaMemcpy(getCudaData(op.getOutput()), getCudaData(op.getInput()),
             bytes, cudaMemcpyDeviceToDevice);

  auto idx_f32 = newCudaData(op.getIndices(), cuda::DT_F32);
  cuda::bmIndexPut(getCudaData(op.getInput()), idx_f32.get(),
                    getCudaData(op.getValues()), getCudaData(op.getOutput()),
                    num_indices, inner_dim, accumulate);
}

void py_cuda::cudaIndexPutOp(tpu::IndexPutOp op) {
  auto in_shape = module::getShape(op.getInput());
  int num_indices = module::getNumElements(op.getIndices());
  int inner_dim = 1;
  for (int i = 1; i < (int)in_shape.size(); i++) inner_dim *= in_shape[i];
  bool accumulate = op.getAccumulate();

  size_t bytes = module::getNumElements(op.getOutput()) * sizeof(float);
  cudaMemcpy(getCudaData(op.getOutput()), getCudaData(op.getInput()),
             bytes, cudaMemcpyDeviceToDevice);

  auto idx_f32 = newCudaData(op.getIndices(), cuda::DT_F32);
  cuda::bmIndexPut(getCudaData(op.getInput()), idx_f32.get(),
                    getCudaData(op.getValues()), getCudaData(op.getOutput()),
                    num_indices, inner_dim, accumulate);
}
