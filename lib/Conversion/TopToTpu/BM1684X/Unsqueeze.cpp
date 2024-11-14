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

void UnsqueezeTryLowering::Lowering(PatternRewriter &rewriter,
                                    top::UnsqueezeOp op) const {

  auto prev_op = op.getInput().getDefiningOp();
  if (!prev_op->hasTrait<trait::ShapeProducer>()) {
    return;
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto v = op.getResult();
  auto shape = module::getShape(v);
  auto ctx = v.getContext();
  Type new_type = RankedTensorType::get(shape, IntegerType::get(ctx, 32));
  rewriter.replaceOpWithNewOp<tpu::ShapeUnsqueezeOp>(op, new_type,
                                                     op.getOperand(), attrs);
}

void UnsqueezeLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::UnsqueezeOp op) const {
  lowering_common_f32<tpu::UnsqueezeOp>(rewriter, op);
}
void UnsqueezeLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::UnsqueezeOp op,
                                     bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void UnsqueezeLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::UnsqueezeOp op,
                                     bool asymmetric) const {
  lowering_common_int8<tpu::UnsqueezeOp>(rewriter, op, asymmetric);
}

void UnsqueezeLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::UnsqueezeOp op) const {
  lowering_common_bf16<tpu::UnsqueezeOp>(rewriter, op);
}

void UnsqueezeLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::UnsqueezeOp op) const {
  lowering_common_f16<tpu::UnsqueezeOp>(rewriter, op);
}

void UnsqueezeLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::UnsqueezeOp op) const {
  bool isE4 = module::getMode() == module::Mode::F8E4M3;
  lowering_common_f8<tpu::UnsqueezeOp>(rewriter, op, isE4);
}

void UnsqueezeLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::UnsqueezeOp op) const {
  lowering_common<tpu::UnsqueezeOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
