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

void py_cuda::cudaShapeCastOp(tpu::ShapeCastOp op) {
  auto num = module::getNumElements(op.getOutput());
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto in_type = getCudaType(op.getInput());
  auto out_type = getCudaType(op.getOutput());
  if (in_type == out_type) {
    auto bytes = num * module::getDtypeSize(op.getOutput());
    CHECK_CUDA(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice));
    return;
  }
  CHECK_CUDA(cuda::convertType(input, output, num, in_type, out_type));
}
