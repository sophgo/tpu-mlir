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

void py_cuda::cudaAddOp(tpu::AddOp op) {
  auto out = op.getOutput();
  if (!module::isUniformQuantized(out)) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto num_inputs = op.getInputs().size();
  if (2 != num_inputs) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto input0 = getCudaData(op.getInputs()[0]);
  auto input1 = getCudaData(op.getInputs()[1]);
  auto output = getCudaData(out);
  auto out_stype = module::getStorageType(out);
  auto out_sign = !out_stype.isUnsignedInteger(8);
  auto num_out = module::getNumElements(out);
  if (module::isCV18xx()) {
    auto multiplier_v =
        module::getI64Array(op.getMultipliers(), op.getInputs().size(), 1);
    auto rshift_v = module::getI64Array(op.getRshifts(), 1, 0);
    cudaAddInt8(input0, input1, output, multiplier_v->at(0),
                multiplier_v->at(1), rshift_v->at(0), num_out);
  } else {
    auto multiplier_v =
        module::getI64Array(op.getMultipliers(), op.getInputs().size(), 1);
    auto rshift_v =
        module::getI64Array(op.getRshifts(), op.getInputs().size(), 0);
    cudaAddInt8(input0, input1, output, multiplier_v->at(0),
                multiplier_v->at(1), rshift_v->at(0), rshift_v->at(1), num_out,
                out_sign);
  }
  if (op.getDoRelu()) {
    cudaRelu(output, num_out, getCudnnType(op.getOutput()));
  }
}
