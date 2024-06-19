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

void py_cuda::cudaAdd(tpu::AddOp op) {
  auto out = op.getOutput();
  if (!module::isCV18xx()) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  if (!module::isUniformQuantized(out)) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto num_inputs = op.getInputs().size();
  if (2 != num_inputs) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto multiplier_v =
      module::getI64Array(op.getMultipliers(), op.getInputs().size(), 1);
  auto rshift_v = module::getI64Array(op.getRshifts(), 1, 0);
  auto input0 = getCudaData(op.getInputs()[0]);
  auto input1 = getCudaData(op.getInputs()[1]);
  auto output = getCudaData(out);
  cudaAddInt8(input0, input1, output, multiplier_v->at(0), multiplier_v->at(1),
              rshift_v->at(0), module::getNumElements(out));
}
