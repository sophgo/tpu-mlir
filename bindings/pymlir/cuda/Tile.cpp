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

void py_cuda::cudaTileOp(tpu::TileOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  std::vector<int64_t> in_shape = module::getShape(op.getInput());
  std::vector<int64_t> out_shape = module::getShape(op.getOutput());
  int out_elems = module::getNumElements(op.getOutput());
  auto num_dims = in_shape.size();
  int64_t *device_in_shape;
  int64_t *device_out_shape;
  cudaMalloc(&device_in_shape, num_dims * sizeof(int64_t));
  cudaMalloc(&device_out_shape, num_dims * sizeof(int64_t));
  cudaMemcpy(device_in_shape, in_shape.data(), num_dims * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_out_shape, out_shape.data(), num_dims * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cuda::tile(input, output, device_in_shape, device_out_shape, num_dims, out_elems, module::getDtypeSize(op.getOutput()));
  cudaFree(device_in_shape);
  cudaFree(device_out_shape);
}
