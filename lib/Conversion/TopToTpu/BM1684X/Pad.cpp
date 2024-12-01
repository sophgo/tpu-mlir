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

void PadLowering::LoweringF32(PatternRewriter &rewriter, top::PadOp op) const {
  auto op_ = op.getOperation();
  auto mode = tpu::symbolizePaddingMode(op.getMode())
                  .value_or(tpu::PaddingMode::constant);
  op_->setAttr("mode", tpu::PaddingModeAttr::get(op.getContext(), mode));
  if (!op.getPaddingsT())
    op->insertOperands(1, {module::getNoneOp(op)});
  lowering_common_f32<tpu::PadOp>(rewriter, op, 5);
}
void PadLowering::LoweringINT4(PatternRewriter &rewriter, top::PadOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void PadLowering::LoweringINT8(PatternRewriter &rewriter, top::PadOp op,
                               bool asymmetric) const {
  int64_t in_zp;
  double in_scale;
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  if (op.getPaddingsT()) {
    operands.push_back(op.getPaddingsT());
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  operands.push_back(module::getNoneOp(op));
  operands.push_back(module::getNoneOp(op));
  operands.push_back(module::getNoneOp(op));
  module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp, asymmetric);

  std::vector<NamedAttribute> attrs;
  auto val = op.getVal().convertToDouble();
  val = std::round(val / in_scale + in_zp);
  attrs.push_back(rewriter.getNamedAttr("paddings", op.getPaddingsAttr()));
  attrs.push_back(rewriter.getNamedAttr("val", rewriter.getF64FloatAttr(val)));
  auto m = tpu::symbolizePaddingMode(op.getMode())
               .value_or(tpu::PaddingMode::constant);
  attrs.push_back(rewriter.getNamedAttr(
      "mode", tpu::PaddingModeAttr::get(op->getContext(), m)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, operands, attrs);
}

void PadLowering::LoweringBF16(PatternRewriter &rewriter, top::PadOp op) const {
  auto op_ = op.getOperation();
  auto mode = tpu::symbolizePaddingMode(op.getMode())
                  .value_or(tpu::PaddingMode::constant);
  op_->setAttr("mode", tpu::PaddingModeAttr::get(op.getContext(), mode));
  if (!op.getPaddingsT())
    op->insertOperands(1, {module::getNoneOp(op)});
  lowering_common_bf16<tpu::PadOp>(rewriter, op, 5);
}

void PadLowering::LoweringF16(PatternRewriter &rewriter, top::PadOp op) const {
  auto op_ = op.getOperation();
  auto mode = tpu::symbolizePaddingMode(op.getMode())
                  .value_or(tpu::PaddingMode::constant);
  op_->setAttr("mode", tpu::PaddingModeAttr::get(op.getContext(), mode));
  if (!op.getPaddingsT())
    op->insertOperands(1, {module::getNoneOp(op)});
  lowering_common_f16<tpu::PadOp>(rewriter, op, 5);
}

void PadLowering::LoweringF8(PatternRewriter &rewriter, top::PadOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void PadLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::PadOp op) const {
  auto op_ = op.getOperation();
  auto mode = tpu::symbolizePaddingMode(op.getMode())
                  .value_or(tpu::PaddingMode::constant);
  op_->setAttr("mode", tpu::PaddingModeAttr::get(op.getContext(), mode));
  if (!op.getPaddingsT())
    op->insertOperands(1, {module::getNoneOp(op)});
  lowering_common<tpu::PadOp>(rewriter, op, op.getOutput().getType(), 5);
}

} // namespace bm1684x
} // namespace tpu_mlir
