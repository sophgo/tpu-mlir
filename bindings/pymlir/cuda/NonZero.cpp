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

void py_cuda::cudaNonZeroOp(tpu::NonZeroOp op) {
  auto shape = module::getShape(op.getInput());
  int dims = shape.size();
  int total = module::getNumElements(op.getInput());
  int order = op.getOrder().str() == "ColMajor" ? 0 : 1;

  // Copy input to host
  std::vector<float> input(total);
  CHECK_CUDA(cudaMemcpy(input.data(), getCudaData(op.getInput()),
                        total * sizeof(float), cudaMemcpyDeviceToHost));

  // Step 1: collect flat indices of non-zero elements (matches CPU reference)
  std::vector<int> indices;
  indices.reserve(total);
  for (int i = 0; i < total; ++i) {
    if (input[i] != 0) {
      indices.push_back(i);
    }
  }
  int pos_num = (int)indices.size();

  // Step 2: decompose flat indices to N-D coordinates (matches CPU reference)
  std::vector<int> coords(pos_num * dims);
  if (dims > 1) {
    for (int i = 0; i < pos_num; ++i) {
      int left = indices[i];
      for (int j = dims - 1; j >= 0; --j) {
        int k = (order == 0) ? (i * dims + j) : (j * pos_num + i);
        coords[k] = (shape[j] == 1) ? 0 : (left % shape[j]);
        left /= shape[j];
      }
    }
  } else {
    for (int i = 0; i < pos_num; ++i) {
      coords[i] = indices[i];
    }
  }

  // Write coordinates to GPU output (int32 matching MLIR i32 type)
  CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), coords.data(),
                        pos_num * dims * sizeof(int),
                        cudaMemcpyHostToDevice));
}
