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

void SqrtLowering::LoweringF32(PatternRewriter &rewriter,
                               top::SqrtOp op) const {
  lowering_common_f32<tpu::SqrtOp>(rewriter, op.getOperation());
}

void SqrtLowering::LoweringINT8(PatternRewriter &rewriter, top::SqrtOp op,
                                bool asymmetric) const {
  auto stype = Module::getStorageType(op.output());
  auto table = create_lookup_table(op.input(), op.output(), asymmetric,
                                   [](double val) { return std::sqrt(val); });
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(
      op, newType,
      ValueRange{op.input(), table, Module::getNoneOp(op.getOperation())},
      attrs);
}

void SqrtLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::SqrtOp op) const {
  LoweringF32(rewriter, op);
}

void SqrtLowering::LoweringF16(PatternRewriter &rewriter,
                               top::SqrtOp op) const {
  LoweringF32(rewriter, op);
}

void SqrtLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::SqrtOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
