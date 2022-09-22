//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

double active_sigmoid(double val) { return 1 / (1 + std::exp(-val)); }

void top::SigmoidOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                           bool asymmetric) {
  auto op = getOperation();
  auto stype = Module::getStorageType(output());
  Value table =
      create_lookup_table(input(), output(), active_sigmoid, asymmetric);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{input(), table}, attrs);
}

void top::SigmoidOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::SigmoidOp>(rewriter, getOperation());
}

void top::SigmoidOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::SigmoidOp, Float32Type>(rewriter, getOperation());
}

void top::SigmoidOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::SigmoidOp, Float32Type>(rewriter, getOperation());
}

void top::SigmoidOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("SigmoidOp not support now");
}
