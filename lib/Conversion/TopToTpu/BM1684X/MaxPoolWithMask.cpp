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

void MaxPoolWithMaskLowering::LoweringF32(PatternRewriter &rewriter,
                                          top::MaxPoolWithMaskOp op) const {
  rewriter.replaceOpWithNewOp<tpu::MaxPoolWithMaskOp>(
      op, op->getResultTypes(), op->getOperands(), op->getAttrs());
}
void MaxPoolWithMaskLowering::LoweringINT4(PatternRewriter &rewriter,
                                           top::MaxPoolWithMaskOp op,
                                           bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void MaxPoolWithMaskLowering::LoweringINT8(PatternRewriter &rewriter,
                                           top::MaxPoolWithMaskOp op,
                                           bool asymmetric) const {
  rewriter.replaceOpWithNewOp<tpu::MaxPoolWithMaskOp>(
      op, op->getResultTypes(), op->getOperands(), op->getAttrs());
}

void MaxPoolWithMaskLowering::LoweringBF16(PatternRewriter &rewriter,
                                           top::MaxPoolWithMaskOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MaxPoolWithMaskLowering::LoweringF16(PatternRewriter &rewriter,
                                          top::MaxPoolWithMaskOp op) const {
  std::vector<Value> operands;
  std::vector<Type> new_types;
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  operands.push_back(op->getOperand(0));

  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  builder.setInsertionPoint(op);

  // for (int i = 0; i < 2; i++) {
  //   auto out = op.getResult(i);
  //   new_types.push_back(getQuantF16Type(out));
  // }
  new_types.push_back(getQuantF16Type(op.getResult(0)));

  auto shape = module::getShape(op.getResult(1));
  auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
  new_types.push_back(new_type);
  rewriter.replaceOpWithNewOp<tpu::MaxPoolWithMaskOp>(op, new_types, operands,
                                                      attrs);
}

void MaxPoolWithMaskLowering::LoweringF8(PatternRewriter &rewriter,
                                         top::MaxPoolWithMaskOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MaxPoolWithMaskLowering::LoweringQuantized(
    PatternRewriter &rewriter, top::MaxPoolWithMaskOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
