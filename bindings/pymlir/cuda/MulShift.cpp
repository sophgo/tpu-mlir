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

void py_cuda::cudaMulShiftOp(tpu::MulShiftOp op) {
  if (!module::isUniformQuantized(op.getInput())) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  auto num = module::getBytes(op.getOutput());
  int32_t m = op.getMultiplier();
  int32_t s = op.getRshift();
  int32_t input_zp = 0, output_zp = 0;
  if (module::isUniformQuantized(op.getInput())) {
    auto qtype = module::getUniformQuantizedType(op.getInput());
    input_zp = qtype.getZeroPoint();
  }
  if (module::isUniformQuantized(op.getOutput())) {
    auto qtype = module::getUniformQuantizedType(op.getOutput());
    output_zp = qtype.getZeroPoint();
  }
  cuda::mulShift(input, output, m, s, num, getCudaType(op.getInput()), input_zp, output_zp);
}
