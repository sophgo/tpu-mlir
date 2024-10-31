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

void ClipTryLowering::Lowering(PatternRewriter &rewriter, top::ClipOp op) const {

  // auto prev_op = op.getInputs().getDefiningOp();
  // if (!prev_op->hasTrait<trait::ShapeProducer>()) {
  //   return;
  // }
  if (!isa_shape_subnet_op(op))
    return;

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("min", rewriter.getF32FloatAttr(op.getMin().convertToDouble())));
  attrs.push_back(rewriter.getNamedAttr("max", rewriter.getF32FloatAttr(op.getMax().convertToDouble())));

  auto v = op.getOutput();
  Type new_type = RankedTensorType::get(module::getShape(v), rewriter.getF32Type() );
  rewriter.replaceOpWithNewOp<tpu::ShapeClipOp>(op, new_type, op->getOperands(), attrs);
}

void ClipLowering::LoweringF32(PatternRewriter &rewriter, top::ClipOp op) const {
  lowering_common_f32<tpu::ClipOp>(rewriter, op);
}
void ClipLowering::LoweringINT4(PatternRewriter &rewriter, top::ClipOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ClipLowering::LoweringINT8(PatternRewriter &rewriter, top::ClipOp op,
                               bool asymmetric) const {
  // nodechip fix8b to be implemented,
  if(module::isMARS3())
    LoweringBF16(rewriter, op);
  else
    LoweringF16(rewriter, op);
}

void ClipLowering::LoweringBF16(PatternRewriter &rewriter, top::ClipOp op) const {
  lowering_common_bf16<tpu::ClipOp>(rewriter, op);
}

void ClipLowering::LoweringF16(PatternRewriter &rewriter, top::ClipOp op) const {
  lowering_common_f16<tpu::ClipOp>(rewriter, op);
}

void ClipLowering::LoweringF8(PatternRewriter &rewriter, top::ClipOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void ClipLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::ClipOp op) const {
  LoweringF16(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir
