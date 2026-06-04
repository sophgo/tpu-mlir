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

void py_cuda::cudaMaxOp(top::MaxOp op) {
  auto inputs = op.getInputs();
  auto a = getCudaData(inputs[0]);
  auto b = getCudaData(inputs[1]);
  auto output = getCudaData(op.getOutput());
  int num = module::getNumElements(op.getOutput());
  cuda::bmMax(a, b, output, num);
}

void py_cuda::cudaMaxOp(tpu::MaxOp op) {
  auto inputs = op.getInputs();
  auto a = getCudaData(inputs[0]);
  auto b = getCudaData(inputs[1]);
  auto output = getCudaData(op.getOutput());
  int num = module::getNumElements(op.getOutput());

  auto stype = module::getStorageType(op.getOutput());
  if (stype.isF32()) {
    cuda::bmMax(a, b, output, num);
  } else {
    auto a_f32 = newCudaData(inputs[0], cuda::DT_F32);
    auto b_f32 = newCudaData(inputs[1], cuda::DT_F32);
    auto output_f32 = cuda_malloc(num * sizeof(float));
    cuda::bmMax(a_f32.get(), b_f32.get(), output_f32.get(), num);
    cuda::convertType(output_f32.get(), output, num, cuda::DT_F32, getCudaType(op.getOutput()));
  }
}
