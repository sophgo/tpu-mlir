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

double active_log(double val) { return std::log(val); }

void top::LogOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                       bool asymmetric) {
  auto ctx = getContext();
  auto op = getOperation();
  auto stype = Module::getStorageType(output());
  Value table = create_lookup_table(input(), output(), active_log, asymmetric);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{input(), table}, attrs);
}

void top::LogOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::LogOp>(rewriter, getOperation());
}

void top::LogOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::LogOp, Float32Type>(rewriter, getOperation());
}

void top::LogOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::LogOp, Float32Type>(rewriter, getOperation());
}

void top::LogOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("LogOp not support now");
}
