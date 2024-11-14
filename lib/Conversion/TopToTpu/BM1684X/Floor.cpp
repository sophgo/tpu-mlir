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

void FloorLowering::LoweringF32(PatternRewriter &rewriter,
                                top::FloorOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::FLOOR));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}
void FloorLowering::LoweringINT4(PatternRewriter &rewriter, top::FloorOp op,
                                 bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void FloorLowering::LoweringINT8(PatternRewriter &rewriter, top::FloorOp op,
                                 bool asymmetric) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    [](double val) { return std::floor(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void FloorLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::FloorOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::FLOOR));
  lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
}

void FloorLowering::LoweringF16(PatternRewriter &rewriter,
                                top::FloorOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::FLOOR));
  lowering_common_f16<tpu::ActiveOp>(rewriter, op_);
}

void FloorLowering::LoweringF8(PatternRewriter &rewriter,
                               top::FloorOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void FloorLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::FloorOp op) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), true,
                                    [](double val) { return std::floor(val); });
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, op.getOutput().getType(),
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684x
} // namespace tpu_mlir
