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

void MaskRCNNBboxPoolerLowering::LoweringF32(
    PatternRewriter &rewriter, top::MaskRCNNBboxPoolerOp op) const {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    auto opd = op->getOperand(i);
    operands.push_back(opd);
  }
  // add buffer
  for (auto i = 0; i < 2; ++i) {
    operands.push_back(module::getNoneOp(op));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  std::vector<Type> new_types;
  new_types.push_back(op.getResultRes().getType());
  new_types.push_back(op.getResultRois().getType());

  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  builder.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<tpu::MaskRCNNBboxPoolerOp>(op, new_types,
                                                         operands, attrs);
  return;
}

void MaskRCNNBboxPoolerLowering::LoweringINT8(PatternRewriter &rewriter,
                                              top::MaskRCNNBboxPoolerOp op,
                                              bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void MaskRCNNBboxPoolerLowering::LoweringINT4(PatternRewriter &rewriter,
                                              top::MaskRCNNBboxPoolerOp op,
                                              bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void MaskRCNNBboxPoolerLowering::LoweringBF16(
    PatternRewriter &rewriter, top::MaskRCNNBboxPoolerOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNBboxPoolerLowering::LoweringF16(
    PatternRewriter &rewriter, top::MaskRCNNBboxPoolerOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNBboxPoolerLowering::LoweringF8(
    PatternRewriter &rewriter, top::MaskRCNNBboxPoolerOp op) const {
  llvm_unreachable("Not Implemented");
}

void MaskRCNNBboxPoolerLowering::LoweringQuantized(
    PatternRewriter &rewriter, top::MaskRCNNBboxPoolerOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
