//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

void top::ConcatOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                          bool asymmetric) {
  auto op = getOperation();
  std::vector<Value> operands;
  for (auto in : inputs()) {
    auto new_in = do_transfer(in, output(), asymmetric);
    operands.push_back(new_in);
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op, newType, operands,
                                             attrs);
}

void top::ConcatOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::ConcatOp>(rewriter, getOperation());
}

void top::ConcatOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::ConcatOp, BFloat16Type>(rewriter, getOperation());
}

void top::ConcatOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::ConcatOp, Float16Type>(rewriter, getOperation());
}

void top::ConcatOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  // if (!output().getType().isUnsignedInteger(8)) {
  //   lowering_common<tpu::ConcatOp>(rewriter, getOperation(),
  //   output().getType());
  // }
  auto op = getOperation();
  std::vector<Value> operands;
  auto out_stype = Module::getStorageType(output());
  if (out_stype.isUnsignedInteger(8)) {
    for (auto in : inputs()) {
      auto new_in = do_transfer_fp(in, output(), true);
      operands.push_back(new_in);
    }
  } else {
    for (auto in : inputs()) {
      operands.push_back(in);
    }
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op, output().getType(),
                                             operands, attrs);
}
