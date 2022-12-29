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

void LogLowering::LoweringF32(PatternRewriter &rewriter, top::LogOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode",
               tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::LN));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void LogLowering::LoweringINT4(PatternRewriter &rewriter, top::LogOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void LogLowering::LoweringINT8(PatternRewriter &rewriter, top::LogOp op,
                               bool asymmetric) const {
  auto stype = module::getStorageType(op.output());
  Value table = create_lookup_table(op.input(), op.output(), asymmetric,
                                    [](double val) { return std::log(val); });
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.input(), table});
}

void LogLowering::LoweringBF16(PatternRewriter &rewriter, top::LogOp op) const {
  LoweringF32(rewriter, op);
}

void LogLowering::LoweringF16(PatternRewriter &rewriter, top::LogOp op) const {
  LoweringF32(rewriter, op);
}

void LogLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::LogOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
