//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"

namespace tpu_mlir {
namespace bm1684x {

void MishLowering::LoweringF32(PatternRewriter &rewriter,
                               top::MishOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::MISH));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void MishLowering::LoweringINT4(PatternRewriter &rewriter, top::MishOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void MishLowering::LoweringINT8(PatternRewriter &rewriter, top::MishOp op,
                                bool asymmetric) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    activate_f(my_mish_activate));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void MishLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::MishOp op) const {
  LoweringF32(rewriter, op);
}

void MishLowering::LoweringF16(PatternRewriter &rewriter,
                               top::MishOp op) const {
  LoweringF32(rewriter, op);
}

void MishLowering::LoweringF8(PatternRewriter &rewriter, top::MishOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void MishLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::MishOp op) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), true,
                                    activate_f(my_mish_activate));
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, op.getOutput().getType(),
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684x
} // namespace tpu_mlir
