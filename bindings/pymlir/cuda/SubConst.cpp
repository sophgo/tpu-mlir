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

void py_cuda::cudaSubConstOp(tpu::SubConstOp op) {
  bool is_reverse = op.getIsReverse();
  if (module::isUniformQuantized(op.getInput())) {
    double const_v = op.getConstVal().convertToDouble();
    auto multi = op.getMultiplier();
    auto shift = op.getRshift();
    int64_t n0, c0, h0, w0;
    module::getNCHW(op.getOutput(), n0, c0, h0, w0, false);
    bool in_signed = !module::getStorageType(op.getInput()).isUnsignedInteger();
    cuda::subConst4DI8(getCudaData(op.getInput()), in_signed, static_cast<int>(const_v), getCudaData(op.getOutput()), op.getDoRelu(), is_reverse,
                  multi, shift, n0, c0, h0, w0);
  } else if (module::getStorageType(op.getInput()).isF32()) {
    double const_v = op.getConstVal().convertToDouble();
    int64_t n0, c0, h0, w0;
    module::getNCHW(op.getOutput(), n0, c0, h0, w0, false);
    cuda::subConst4DF32(getCudaData(op.getInput()), const_v, getCudaData(op.getOutput()), op.getDoRelu(), is_reverse,
                  n0, c0, h0, w0);
  } else {
    double const_v = op.getConstVal().convertToDouble();
    int64_t n0, c0, h0, w0;
    module::getNCHW(op.getOutput(), n0, c0, h0, w0, false);
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto output_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
    cuda::subConst4DF32(input_f32.get(), const_v, output_f32.get(), op.getDoRelu(), is_reverse,
                  n0, c0, h0, w0);
    cuda::convertType(output_f32.get(), getCudaData(op.getOutput()), module::getNumElements(op.getOutput()), cuda::DT_F32,
                  getCudaType(op.getOutput()));
    input_f32.reset();
    output_f32.reset();
  }
}

void py_cuda::cudaSubConstOp(top::SubConstOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  double const_v = op.getConstVal().convertToDouble();
  bool is_reverse = op.getIsReverse();
  int64_t n0, c0, h0, w0;
  module::getNCHW(op.getOutput(), n0, c0, h0, w0, false);
  cuda::subConst4DF32(input, const_v, output, op.getDoRelu(), is_reverse,
                  n0, c0, h0, w0);
}
