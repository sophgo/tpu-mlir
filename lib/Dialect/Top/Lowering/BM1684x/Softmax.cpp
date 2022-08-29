//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
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
  llvm_unreachable("to be supported for Softmax int8 quantize lowering");
}

Value top::SoftmaxOp::lowering_f32_bm1684x() {
  auto op = getOperation();
  OpBuilder builder(getContext());
  std::vector<Value> operands;
  operands.push_back(input());
  auto none = Module::getNoneOp(op);
  operands.push_back(none);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  builder.setInsertionPointAfter(op);
  auto newOp = builder.create<tpu::SoftmaxOp>(op->getLoc(), output().getType(),
                                              operands, attrs);
  return newOp.output();
}

Value top::SoftmaxOp::lowering_bf16_bm1684x() {
  // return lowering_common_float<tpu::SoftmaxOp, BFloat16Type>(getOperation());
  return lowering_f32_bm1684x();
}

Value top::SoftmaxOp::lowering_f16_bm1684x() {
  // return lowering_common_float<tpu::SoftmaxOp, Float16Type>(getOperation());
  return lowering_f32_bm1684x();
}

Value top::SoftmaxOp::lowering_quant_bm1684x() {
  if (Quant::isUniformQuantized(input(), output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  auto op = getOperation();
  OpBuilder builder(getContext());
  const int nInputs = op->getNumOperands();
  int64_t zeropoint;
  double i_scale;
  Quant::getScaleAndZeroPoint(input(), i_scale, zeropoint, true);
  std::vector<float> table(256, 0.0f);
  auto beta_v = beta().convertToDouble();
  auto scale = -i_scale * beta_v;
  for (int i = 0; i < 256; ++i) {
    table[i] = std::exp(scale * i);
  }
  auto table_opd = create_lookup_table(op, table);

  builder.setInsertionPointAfter(op);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp = builder.create<tpu::SoftmaxOp>(
      op->getLoc(), output().getType(), ValueRange{input(), table_opd}, attrs);
  return newOp.output();
}
