//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void ConcatLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::ConcatOp op) const {
  lowering_common_f32<tpu::ConcatOp>(rewriter, op);
}

void ConcatLowering::LoweringINT8(PatternRewriter &rewriter,
                                  top::ConcatOp concatOp,
                                  bool asymmetric) const {
  // checkout whether weight exist
  for (auto in : concatOp.inputs()) {
    if (isa<top::WeightOp>(in.getDefiningOp())) {
      LoweringF16(rewriter, concatOp);
      return;
    }
  }
  auto op = concatOp.getOperation();
  std::vector<Value> operands;
  for (auto in : concatOp.inputs()) {
    auto new_in = do_transfer(in, concatOp.output(), asymmetric);
    operands.push_back(new_in);
  }
  auto newType = getQuantInt8Type(concatOp.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op, newType, operands,
                                             op->getAttrs());
}

void ConcatLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::ConcatOp concatOp) const {
  lowering_common_bf16<tpu::ConcatOp>(rewriter, concatOp.getOperation());
}

void ConcatLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::ConcatOp concatOp) const {
  lowering_common_f16<tpu::ConcatOp>(rewriter, concatOp.getOperation());
}

void ConcatLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::ConcatOp concatOp) const {
  auto op = concatOp.getOperation();
  std::vector<Value> operands;
  auto out_stype = Module::getStorageType(concatOp.output());
  if (out_stype.isUnsignedInteger(8)) {
    for (auto in : concatOp.inputs()) {
      auto new_in = do_transfer_fp(in, concatOp.output(), true);
      operands.push_back(new_in);
    }
  } else {
    for (auto in : concatOp.inputs()) {
      operands.push_back(in);
    }
  }

  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op, concatOp.output().getType(),
                                             operands, op->getAttrs());
}

} // namespace bm1684x
} // namespace tpu_mlir
