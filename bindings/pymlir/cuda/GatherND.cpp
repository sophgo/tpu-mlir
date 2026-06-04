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

void py_cuda::cudaGatherNDOp(top::GatherNDOp op) {
  auto in_shape = module::getShape(op.getInput());
  auto idx_shape = module::getShape(op.getIndices());
  int in_rank = in_shape.size();
  int idx_rank = idx_shape.size();
  int batch_dims = op.getBatchDims();
  int coord_dim = idx_shape[idx_rank - 1];

  // Input strides (up to 8D)
  int in_strides_arr[8] = {};
  int stride = 1;
  for (int d = in_rank - 1; d >= 0; d--) {
    in_strides_arr[d] = stride;
    stride *= in_shape[d];
  }

  // Indices strides (for decoding output coordinates)
  int idx_strides_arr[8] = {};
  stride = 1;
  for (int d = idx_rank - 2; d >= 0; d--) { // last dim is coord, skip
    idx_strides_arr[d] = stride;
    stride *= idx_shape[d];
  }

  // output_total = product of indices dims (minus last coord dim)
  int out_total = 1;
  for (int d = batch_dims; d < idx_rank - 1; d++) out_total *= idx_shape[d];
  for (int d = 0; d < batch_dims; d++) out_total *= idx_shape[d];

  // copy_len = product of input dims after the indexed dims
  int copy_len = 1;
  for (int d = batch_dims + coord_dim; d < in_rank; d++)
    copy_len *= in_shape[d];

  cuda_ptr idx_f32;
  void *idx_data = getCudaData(op.getIndices());
  if (getCudaType(op.getIndices()) != cuda::DT_F32) {
    idx_f32 = newCudaData(op.getIndices(), cuda::DT_F32);
    idx_data = idx_f32.get();
  }
  cuda::bmGatherND(getCudaData(op.getInput()), idx_data,
                    getCudaData(op.getOutput()),
                    (int*)in_shape.data(), in_strides_arr,
                    (int*)idx_shape.data(), idx_strides_arr,
                    batch_dims, idx_rank, coord_dim,
                    out_total, copy_len);
}

void py_cuda::cudaGatherNDOp(tpu::GatherNDOp op) {
  auto in_shape = module::getShape(op.getInputData());
  auto idx_shape = module::getShape(op.getIndices());
  int in_rank = in_shape.size();
  int idx_rank = idx_shape.size();
  int batch_dims = op.getBatchDims();
  int coord_dim = idx_shape[idx_rank - 1];

  int in_strides_arr[8] = {};
  int stride = 1;
  for (int d = in_rank - 1; d >= 0; d--) {
    in_strides_arr[d] = stride;
    stride *= in_shape[d];
  }

  int idx_strides_arr[8] = {};
  stride = 1;
  for (int d = idx_rank - 2; d >= 0; d--) {
    idx_strides_arr[d] = stride;
    stride *= idx_shape[d];
  }

  int out_total = 1;
  for (int d = batch_dims; d < idx_rank - 1; d++) out_total *= idx_shape[d];
  for (int d = 0; d < batch_dims; d++) out_total *= idx_shape[d];

  int copy_len = 1;
  for (int d = batch_dims + coord_dim; d < in_rank; d++)
    copy_len *= in_shape[d];

  cuda_ptr idx_f32;
  void *idx_data = getCudaData(op.getIndices());
  if (getCudaType(op.getIndices()) != cuda::DT_F32) {
    idx_f32 = newCudaData(op.getIndices(), cuda::DT_F32);
    idx_data = idx_f32.get();
  }
  cuda::bmGatherND(getCudaData(op.getInputData()), idx_data,
                    getCudaData(op.getOutput()),
                    (int*)in_shape.data(), in_strides_arr,
                    (int*)idx_shape.data(), idx_strides_arr,
                    batch_dims, idx_rank, coord_dim,
                    out_total, copy_len);
}
