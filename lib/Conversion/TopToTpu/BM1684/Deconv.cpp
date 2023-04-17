//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void DeconvLowering::LoweringINT8(PatternRewriter &rewriter, top::DeconvOp op,
                                  bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}


void DeconvLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::DeconvOp op) const {
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !module::isNone(op.getBias());
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  if (op.getKernelShape().size() == 2) {
    rewriter.replaceOpWithNewOp<tpu::DeconvOp>(op, op.getOutput().getType(),
                                              operands, attrs);
  } else {
    llvm_unreachable("Not Implemented");
  }

}

} // namespace bm1684
} // namespace tpu_mlir
