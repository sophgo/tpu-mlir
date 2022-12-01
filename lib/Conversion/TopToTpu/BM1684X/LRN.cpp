//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Conversion/TopToTpu/TopLowering.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

namespace tpu_mlir {
namespace bm1684x {

void LRNLowering::LoweringF32(PatternRewriter &rewriter, top::LRNOp op) const {
  std::vector<Value> operands;
  operands.push_back(op.input());
  auto none = Module::getNoneOp(op);
  for(int i = 0; i < 2; i++) {
    operands.push_back(none);
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  rewriter.replaceOpWithNewOp<tpu::LRNOp>(op, op.output().getType(),
                                              operands, attrs);
}

void LRNLowering::LoweringINT8(PatternRewriter &rewriter, top::LRNOp op,
                               bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void LRNLowering::LoweringBF16(PatternRewriter &rewriter, top::LRNOp op) const {
  LoweringF32(rewriter, op);
}

void LRNLowering::LoweringF16(PatternRewriter &rewriter, top::LRNOp op) const {
  LoweringF32(rewriter, op);
}

void LRNLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::LRNOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
