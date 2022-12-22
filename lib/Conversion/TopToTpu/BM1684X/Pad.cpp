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
  lowering_common_f32<tpu::PadOp>(rewriter, op, 3);
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
  operands.push_back(op.input());
  operands.push_back(Module::getNoneOp(op));
  operands.push_back(Module::getNoneOp(op));
  Quant::getScaleAndZeroPoint(op.input(), in_scale, in_zp, asymmetric);

  std::vector<NamedAttribute> attrs;
  auto val = op.val().convertToDouble();
  val = std::round(val / in_scale + in_zp);
  attrs.push_back(rewriter.getNamedAttr("paddings", op.paddingsAttr()));
  attrs.push_back(rewriter.getNamedAttr("val", rewriter.getF64FloatAttr(val)));
  attrs.push_back(rewriter.getNamedAttr("mode", op.modeAttr()));

  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, operands, attrs);
}

void PadLowering::LoweringBF16(PatternRewriter &rewriter, top::PadOp op) const {
  lowering_common_bf16<tpu::PadOp>(rewriter, op, 3);
}

void PadLowering::LoweringF16(PatternRewriter &rewriter, top::PadOp op) const {
  lowering_common_f16<tpu::PadOp>(rewriter, op, 3);
}

void PadLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::PadOp op) const {
  lowering_common<tpu::PadOp>(rewriter, op, op.output().getType(), 3);
}

} // namespace bm1684x
} // namespace tpu_mlir
