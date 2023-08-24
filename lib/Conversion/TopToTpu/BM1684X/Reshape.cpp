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

void ReshapeTryLowering::Lowering(PatternRewriter &rewriter,
                                  top::ReshapeOp op) const {
  if (!op.getShapeT() ||
      (op.getShapeT() &&
       !op.getShapeT().getDefiningOp()->hasTrait<trait::ShapeProducer>()))
    return;
  try_insert_device2host(op.getOperation(), 1);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto v = op.getResult();
  auto shape = module::getShape(v);
  auto ctx = v.getContext();
  Type new_type = RankedTensorType::get(shape, IntegerType::get(ctx, 32));
  rewriter.replaceOpWithNewOp<tpu::ShapeAssignOp>(op, new_type,
                                                  op.getOperands(), attrs);
}

void ReshapeLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::ReshapeOp op) const {
  lowering_common_f32<tpu::ReshapeOp>(rewriter, op);
}

void ReshapeLowering::LoweringINT8(PatternRewriter &rewriter, top::ReshapeOp op,
                                   bool asymmetric) const {
  lowering_common_int8<tpu::ReshapeOp>(rewriter, op, asymmetric);
}
void ReshapeLowering::LoweringINT4(PatternRewriter &rewriter, top::ReshapeOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ReshapeLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::ReshapeOp op) const {
  lowering_common_bf16<tpu::ReshapeOp>(rewriter, op);
}

void ReshapeLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::ReshapeOp op) const {
  lowering_common_f16<tpu::ReshapeOp>(rewriter, op);
}

void ReshapeLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::ReshapeOp op) const {
  lowering_common<tpu::ReshapeOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
