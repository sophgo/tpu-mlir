//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

Value top::SoftmaxOp::lowering_int8_bm1684x(bool asymmetric) {
  return lowering_common_float<tpu::SoftmaxOp>(
      getOperation()); // skip int8 quant for now
}

Value top::SoftmaxOp::lowering_f32_bm1684x() {
  return lowering_common_float<tpu::SoftmaxOp>(getOperation());
}

Value top::SoftmaxOp::lowering_bf16_bm1684x() {
  llvm_unreachable("to be supported for Softmax bf16 quantize lowering");
}

Value top::SoftmaxOp::lowering_f16_bm1684x() {
  llvm_unreachable("to be supported for Softmax f16 quantize lowering");
}

Value top::SoftmaxOp::lowering_quant_bm1684x() {
  if (Quant::isUniformQuantized(input(), output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  return lowering_common<tpu::SoftmaxOp>(getOperation(), output().getType());
  // // use f32
  // Builder builder(getContext());
  // auto in_f32 = do_cast(input(), builder.getF32Type(), false);
  // auto op = getOperation();
  // op->setOperand(0, in_f32);
  // auto type = output().getType();
  // auto v =  lowering_common_float<tpu::SoftmaxOp>(op);
  // return do_cast(v, type, true);
}
