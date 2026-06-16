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

void py_cuda::cudaShapeOp(top::ShapeOp op) {
  auto input = op.getInput();
  auto output = op.getOutput();
  auto shape = module::getShape(input);
  auto num_dims = shape.size();
  auto num_elements = module::getNumElements(output);
  // 输出应该是一维，长度等于输入维度数
  assert(num_elements == num_dims);

  auto out_type = module::getStorageType(output);
  void* output_ptr = getCudaData(output);
  if (out_type.isInteger(64)) {
    std::vector<int64_t> shape_vals(num_dims);
    for (size_t i = 0; i < num_dims; ++i) {
      shape_vals[i] = static_cast<int64_t>(shape[i]);
    }
    CHECK_CUDA(cudaMemcpy(output_ptr, shape_vals.data(),
                          num_dims * sizeof(int64_t), cudaMemcpyHostToDevice));
  } else if (out_type.isInteger(32)) {
    std::vector<int32_t> shape_vals(num_dims);
    for (size_t i = 0; i < num_dims; ++i) {
      shape_vals[i] = static_cast<int32_t>(shape[i]);
    }
    CHECK_CUDA(cudaMemcpy(output_ptr, shape_vals.data(),
                          num_dims * sizeof(int32_t), cudaMemcpyHostToDevice));
  } else if (out_type.isF32()) {
    std::vector<float> shape_vals(num_dims);
    for (size_t i = 0; i < num_dims; ++i) {
      shape_vals[i] = static_cast<float>(shape[i]);
    }
    CHECK_CUDA(cudaMemcpy(output_ptr, shape_vals.data(),
                          num_dims * sizeof(float), cudaMemcpyHostToDevice));
  } else {
    UNREACHABLE_OP("Unsupported output type for ShapeOp", op);
  }
}
