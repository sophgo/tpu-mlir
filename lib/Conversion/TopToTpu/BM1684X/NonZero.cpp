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
static void LoweringNonZero(PatternRewriter &rewriter, top::NonZeroOp from,
                            int num_operands) {
  std::vector<Value> operands;
  assert(from->getNumOperands() == 1);
  for (int i = 0; i < from->getNumOperands(); ++i) {
    auto in = from.getOperand();
    operands.push_back(in);
  }
  assert(num_operands == 2);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : from->getAttrs()) {
    attrs.push_back(attr);
  }
  if (num_operands > from->getNumOperands()) {
    auto noneOp = module::getNoneOp(from);
    for (int i = from->getNumOperands(); i < num_operands; i++) {
      operands.push_back(noneOp);
    }
  }
  auto v = from.getResult();
  auto shape = module::getShape(v);
  auto ctx = v.getContext();
  Type new_type = RankedTensorType::get(shape, IntegerType::get(ctx, 32));
  rewriter.replaceOpWithNewOp<tpu::NonZeroOp>(from, new_type, operands, attrs);
}

void NonZeroLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::NonZeroOp op) const {
  LoweringNonZero(rewriter, op, 2);
}

void NonZeroLowering::LoweringINT8(PatternRewriter &rewriter, top::NonZeroOp op,
                                   bool asymmetric) const {
  LoweringNonZero(rewriter, op, 2);
}

void NonZeroLowering::LoweringINT4(PatternRewriter &rewriter, top::NonZeroOp op,
                                   bool asymmetric) const {
  LoweringNonZero(rewriter, op, 2);
}

void NonZeroLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::NonZeroOp op) const {
  LoweringNonZero(rewriter, op, 2);
}

void NonZeroLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::NonZeroOp op) const {

  LoweringNonZero(rewriter, op, 2);
}

void NonZeroLowering::LoweringF8(PatternRewriter &rewriter,
                                 top::NonZeroOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void NonZeroLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::NonZeroOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
