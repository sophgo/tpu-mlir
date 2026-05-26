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

void py_cuda::cudaMaskedFillOp(top::MaskedFillOp op) {
  auto cond = getCudaData(op.getCond());
  auto brn = getCudaData(op.getBrn());
  auto output = getCudaData(op.getOutput());
  float const_val = op.getConstVal().convertToDouble();
  bool inversed = op.getInversed();
  int num_elems = module::getNumElements(op.getOutput());

  cuda::maskedFill(cond, brn, output, const_val, inversed, num_elems);
}

void py_cuda::cudaMaskedFillOp(tpu::MaskedFillOp op) {
  auto cond = getCudaData(op.getCond());
  auto brn = getCudaData(op.getBrn());
  auto output = getCudaData(op.getOutput());
  float const_val = op.getConstVal().convertToDouble();
  bool inversed = op.getInversed();
  int num_elems = module::getNumElements(op.getOutput());

  auto stype = module::getStorageType(op.getOutput());
  if (stype.isF32()) {
    cuda::maskedFill(cond, brn, output, const_val, inversed, num_elems);
  } else {
    auto cond_f32 = newCudaData(cond, num_elems, getCudaType(op.getCond()), cuda::DT_F32);
    auto brn_f32  = newCudaData(brn, module::getNumElements(op.getBrn()), getCudaType(op.getBrn()), cuda::DT_F32);
    auto output_f32 = cuda_malloc(num_elems * sizeof(float));
    cuda::maskedFill(cond_f32.get(), brn_f32.get(), output_f32.get(), const_val, inversed, num_elems);
    cuda::convertType(output_f32.get(), output, num_elems, cuda::DT_F32, getCudaType(op.getOutput()));
  }
}
