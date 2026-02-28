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

void py_cuda::cudaMulConstOp(tpu::MulConstOp op) {
  double const_v = op.getConstVal().convertToDouble();
  auto out_shape = std::vector<int64_t>(module::getShape(op.getOutput()));
  while (out_shape.size() < 6) {
    out_shape.emplace(out_shape.begin(), 1);
  }

  if (module::isUniformQuantized(op.getInput())) {
    cuda::mulShift(getCudaData(op.getInput()), getCudaData(op.getOutput()), op.getMultiplier(), op.getRshift(), module::getNumElements(op.getInput()), getCudaType(op.getOutput()));
  } else if (module::getStorageType(op.getInput()).isF32()) {
    cuda::mulConst6DF32(getCudaData(op.getInput()), const_v, getCudaData(op.getOutput()), op.getDoRelu(),
                  out_shape[0], out_shape[1], out_shape[2], out_shape[3], out_shape[4], out_shape[5]);
  } else if (module::getStorageType(op.getInput()).isF16() || module::getStorageType(op.getInput()).isBF16()){
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto output_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
    cuda::mulConst6DF32(input_f32.get(), const_v, output_f32.get(), op.getDoRelu(),
                  out_shape[0], out_shape[1], out_shape[2], out_shape[3], out_shape[4], out_shape[5]);
    cuda::convertType(output_f32.get(), getCudaData(op.getOutput()), module::getNumElements(op.getOutput()), cuda::DT_F32,
                  getCudaType(op.getOutput()));
    input_f32.reset();
    output_f32.reset();
  } else if (module::getStorageType(op.getInput()).isFloat8E4M3FN()) {
    // maybe used as dequant of fp8 or in fp8 mulconst
    const_v = F16(const_v, true);
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    if (module::getStorageType(op.getOutput()).isF32()) {
      cuda::mulConst6DF32(input_f32.get(), const_v, getCudaData(op.getOutput()), op.getDoRelu(),
                  out_shape[0], out_shape[1], out_shape[2], out_shape[3], out_shape[4], out_shape[5]);
    } else {
      auto output_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
      cuda::mulConst6DF32(input_f32.get(), const_v, output_f32.get(), op.getDoRelu(),
                  out_shape[0], out_shape[1], out_shape[2], out_shape[3], out_shape[4], out_shape[5]);
      if (module::getStorageType(op.getOutput()).isFloat8E4M3FN()){
        cuda::requantF8(output_f32.get(), getCudaData(op.getOutput()), 1.0 , out_shape[0], out_shape[1], out_shape[2], out_shape[3], out_shape[4], out_shape[5], op.getDoRelu());
      } else if (module::getStorageType(op.getOutput()).isF16() || module::getStorageType(op.getOutput()).isBF16()) {
        cuda::convertType(output_f32.get(), getCudaData(op.getOutput()), module::getNumElements(op.getOutput()), cuda::DT_F32,
                        getCudaType(op.getOutput()));
      }
      output_f32.reset();
    }
    input_f32.reset();
  }
}

void py_cuda::cudaMulConstOp(top::MulConstOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  double const_v = op.getConstVal().convertToDouble();
  auto out_shape = std::vector<int64_t>(module::getShape(op.getOutput()));
  while (out_shape.size() < 6) {
    out_shape.emplace(out_shape.begin(), 1);
  }
  cuda::mulConst6DF32(input, const_v, output, op.getDoRelu(),
                  out_shape[0], out_shape[1], out_shape[2], out_shape[3], out_shape[4], out_shape[5]);
}
