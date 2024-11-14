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

void MaskRCNNGetBboxBLowering::LoweringF32(PatternRewriter &rewriter,
                                           top::MaskRCNNGetBboxBOp op) const {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    auto opd = op->getOperand(i);
    operands.push_back(opd);
  }
  // add buffer
  for (auto i = 0; i < 20; ++i) {
    operands.push_back(module::getNoneOp(op));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  std::vector<Type> new_types;
  new_types.push_back(op.getResultDetBboxes().getType());
  if (!module::isNone(op.getResultDetLabels())) {
    auto shape = module::getShape(op.getResultDetLabels());
    auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
    new_types.push_back(new_type);
  } else {
    new_types.push_back(op.getResultDetLabels().getType());
  }

  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  builder.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<tpu::MaskRCNNGetBboxBOp>(op, new_types, operands,
                                                       attrs);
  return;
}

void MaskRCNNGetBboxBLowering::LoweringINT4(PatternRewriter &rewriter,
                                            top::MaskRCNNGetBboxBOp op,
                                            bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNGetBboxBLowering::LoweringINT8(PatternRewriter &rewriter,
                                            top::MaskRCNNGetBboxBOp op,
                                            bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNGetBboxBLowering::LoweringBF16(PatternRewriter &rewriter,
                                            top::MaskRCNNGetBboxBOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNGetBboxBLowering::LoweringF16(PatternRewriter &rewriter,
                                           top::MaskRCNNGetBboxBOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNGetBboxBLowering::LoweringF8(PatternRewriter &rewriter,
                                          top::MaskRCNNGetBboxBOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNGetBboxBLowering::LoweringQuantized(
    PatternRewriter &rewriter, top::MaskRCNNGetBboxBOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
