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

void SigmoidLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::SigmoidOp op) const {
  lowering_common_float<tpu::SigmoidOp>(rewriter, op);
}

static double active_sigmoid(double val) { return 1 / (1 + std::exp(-val)); }

void SigmoidLowering::LoweringINT8(PatternRewriter &rewriter,
                                   top::SigmoidOp op, bool asymmetric) const {
  auto stype = Module::getStorageType(op.output());
  Value table =
      create_lookup_table(op.input(), op.output(), active_sigmoid, asymmetric);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.input(), table}, attrs);
}

void SigmoidLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::SigmoidOp op) const {
  lowering_common_float<tpu::SigmoidOp, BFloat16Type>(rewriter, op);
}

void SigmoidLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::SigmoidOp op) const {
  lowering_common_float<tpu::SigmoidOp, Float16Type>(rewriter, op);
}

void SigmoidLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::SigmoidOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
