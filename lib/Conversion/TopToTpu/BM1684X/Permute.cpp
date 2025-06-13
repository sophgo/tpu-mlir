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

void PermuteTryLowering::Lowering(PatternRewriter &rewriter,
                                  top::PermuteOp op) const {
  if (!isa_shape_subnet_op(op))
    return;

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  auto v = op.getResult();
  auto shape = module::getShape(v);
  auto ctx = v.getContext();
  Type new_type = RankedTensorType::get(shape, IntegerType::get(ctx, 32));
  rewriter.replaceOpWithNewOp<tpu::ShapeTransposeOp>(op, new_type,
                                                     op.getOperand(), attrs);
}

void PermuteLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::PermuteOp op) const {
  lowering_common_f32<tpu::PermuteOp>(rewriter, op, 2);
}
void PermuteLowering::LoweringINT4(PatternRewriter &rewriter, top::PermuteOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void PermuteLowering::LoweringINT8(PatternRewriter &rewriter, top::PermuteOp op,
                                   bool asymmetric) const {
  lowering_common_int8<tpu::PermuteOp>(rewriter, op, asymmetric, 2);
}

void PermuteLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::PermuteOp op) const {
  lowering_common_bf16<tpu::PermuteOp>(rewriter, op, 2);
}

void PermuteLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::PermuteOp op) const {
  lowering_common_f16<tpu::PermuteOp>(rewriter, op, 2);
}

void PermuteLowering::LoweringF8(PatternRewriter &rewriter,
                                 top::PermuteOp op) const {
  bool isE4 = module::getMode() == module::Mode::F8E4M3;
  lowering_common_f8<tpu::PermuteOp>(rewriter, op, isE4, 2);
}

void PermuteLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::PermuteOp op) const {
  lowering_common<tpu::PermuteOp>(rewriter, op, op.getOutput().getType(), 2);
}

} // namespace bm1684x
} // namespace tpu_mlir
