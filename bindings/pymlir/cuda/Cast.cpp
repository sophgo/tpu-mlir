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

void py_cuda::cudaCast(tpu::CastOp op) {
  auto num_elem = module::getNumElements(op.getOutput());
  auto in_type = module::getStorageType(op.getInput());
  auto out_type = module::getStorageType(op.getOutput());
  bool isInQuant = module::isUniformQuantized(op.getInput());
  bool isOutQuant = module::isUniformQuantized(op.getOutput());
  bool fInput = in_type.isIntOrIndex() == false;
  bool fOutput = out_type.isIntOrIndex() == false;
  bool is_cv18xx = module::isCV18xx();
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  if (isOutQuant && fInput) {
    //auto qtype = module::getUniformQuantizedType(op.getOutput());
  } else if (fOutput && isInQuant) {
    auto qtype = module::getUniformQuantizedType(op.getInput());
    auto scale = qtype.getScale();
    if (is_cv18xx) {
      cudaCVScaleToF32(input, output, BF16(scale), num_elem);
    } else {
      cudaScaleToF32(input, output, scale, num_elem);
    }
    return;
  }
  UNREACHABLE_OP("Not Implemented", op);
}
