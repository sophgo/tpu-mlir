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

void py_cuda::cudaMulOp(tpu::MulOp op) {
  auto out = op.getOutput();
  if (!module::isUniformQuantized(out)) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto num_inputs = op.getInputs().size();
  if (2 != num_inputs) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto in0 = op.getInputs()[0];
  auto in1 = op.getInputs()[1];
  auto input0 = getCudaData(in0);
  auto input1 = getCudaData(in1);
  auto output = getCudaData(out);
  auto num_out = module::getNumElements(out);
  int multiplier = op.getMultiplier();
  int rshift = op.getRshift();
  bool qdm = op.getQuantMode() == tpu::RequantMode::QDM;
  bool sign0 = !module::getStorageType(in0).isUnsignedInteger(8);
  bool sign1 = !module::getStorageType(in1).isUnsignedInteger(8);
  bool sign2 = !module::getStorageType(out).isUnsignedInteger(8);
  cudaMulInt8(input0, input1, output, sign0, sign1, sign2, multiplier, rshift,
              num_out, qdm, op.getDoRelu());
}
