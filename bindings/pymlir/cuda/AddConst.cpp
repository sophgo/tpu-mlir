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

void py_cuda::cudaAddConstOp(tpu::AddConstOp op) {
  float const_v = op.getConstVal().convertToDouble();
  int64_t n, c, h, w;
  module::getNCHW(op.getOutput(), n, c, h, w, false);
  if (module::getStorageType(op.getInput()).isF32()) {
    cuda::addConst4DF32(getCudaData(op.getInput()), const_v,
                        getCudaData(op.getOutput()), op.getDoRelu(), n, c, h, w);
  } else {
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto output_f32 = cuda_malloc(module::getNumElements(op.getOutput()) * sizeof(float));
    cuda::addConst4DF32(input_f32.get(), const_v, output_f32.get(),
                        op.getDoRelu(), n, c, h, w);
    cuda::convertType(output_f32.get(), getCudaData(op.getOutput()),
                      module::getNumElements(op.getOutput()), cuda::DT_F32,
                      getCudaType(op.getOutput()));
    input_f32.reset();
    output_f32.reset();
  }
}

void py_cuda::cudaAddConstOp(top::AddConstOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  float const_v = op.getConstVal().convertToDouble();
  int64_t n, c, h, w;
  module::getNCHW(op.getOutput(), n, c, h, w, false);
  cuda::addConst4DF32(input, const_v, output, op.getDoRelu(), n, c, h, w);
}
