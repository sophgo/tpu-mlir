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

void py_cuda::cudaShapeOp(tpu::ShapeOp op) {
  auto shape = module::getShape(op.getInput());
  auto num = module::getNumElements(op.getOutput());
  std::vector<int32_t> shape_i32(num);
  for (size_t i = 0; i < shape.size() && i < (size_t)num; ++i)
    shape_i32[i] = (int32_t)shape[i];
  CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), shape_i32.data(),
                        num * sizeof(int32_t), cudaMemcpyHostToDevice));
}
