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

static void LoweringReshapeToShapeAssign(PatternRewriter &rewriter,
                                         top::ReshapeOp op, Type type) {
  try_insert_device2host(op.getOperation(), 1);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  rewriter.replaceOpWithNewOp<tpu::ShapeAssignOp>(op, type, op.getOperands(),
                                                  attrs);
}

static bool check_unlowering(mlir::Operation *op) {
  if (!(module::isBM1688() || module::isSG2380() || module::isMARS3() ||
        module::isSGTPUV8()))
    return false;

  for (auto &use : cast<top::ReshapeOp>(op).getOutput().getUses()) {
    if (isa<top::MatMulOp, tpu::MatMulOp>(use.getOwner()) &&
        use.getOperandNumber() == 2) {
      return true;
    }
  }
  return false;
}

void ReshapeTryLowering::Lowering(PatternRewriter &rewriter,
                                  top::ReshapeOp op) const {
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
  rewriter.replaceOpWithNewOp<tpu::ShapeReshapeOp>(op, new_type,
                                                   op.getOperands(), attrs);
}

void ReshapeLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::ReshapeOp op) const {
  if (op.getShapeT()) {
    auto new_type = getQuantFloatType(op->getResult(0));
    LoweringReshapeToShapeAssign(rewriter, op, new_type);
  } else {
    lowering_common_f32<tpu::ReshapeOp>(rewriter, op);
  }
}

void ReshapeLowering::LoweringINT8(PatternRewriter &rewriter, top::ReshapeOp op,
                                   bool asymmetric) const {
  if (op.getShapeT()) {
    auto new_type = getQuantInt8Type(op->getResult(0), asymmetric);
    LoweringReshapeToShapeAssign(rewriter, op, new_type);
  } else {
    lowering_common_int8<tpu::ReshapeOp>(rewriter, op, asymmetric);
  }
}
void ReshapeLowering::LoweringINT4(PatternRewriter &rewriter, top::ReshapeOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ReshapeLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::ReshapeOp op) const {
  if (op.getShapeT()) {
    auto new_type = getQuantFloatType<mlir::BFloat16Type>(op->getResult(0));
    LoweringReshapeToShapeAssign(rewriter, op, new_type);
  } else {
    if (check_unlowering(op.getOperation()))
      // trick for A2
      lowering_common_f32<tpu::ReshapeOp>(rewriter, op);
    else
      lowering_common_bf16<tpu::ReshapeOp>(rewriter, op);
  }
}

void ReshapeLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::ReshapeOp op) const {
  if (op.getShapeT()) {
    auto new_type = getQuantFloatType<mlir::Float16Type>(op->getResult(0));
    LoweringReshapeToShapeAssign(rewriter, op, new_type);
  } else {
    if (check_unlowering(op.getOperation()))
      // trick for A2
      lowering_common_f32<tpu::ReshapeOp>(rewriter, op);
    else
      lowering_common_f16<tpu::ReshapeOp>(rewriter, op);
  }
}

void ReshapeLowering::LoweringF8(PatternRewriter &rewriter,
                                 top::ReshapeOp op) const {
  bool isE4 = module::getMode() == module::Mode::F8E4M3;
  if (op.getShapeT()) {
    Type new_type;
    if (isE4)
      new_type = getQuantF8E4M3Type(op->getResult(0));
    else
      new_type = getQuantF8E5M2Type(op->getResult(0));
    LoweringReshapeToShapeAssign(rewriter, op, new_type);
  } else {
    lowering_common_f8<tpu::ReshapeOp>(rewriter, op, isE4);
  }
}

void ReshapeLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::ReshapeOp op) const {
  if (op.getShapeT()) {
    auto new_type = op->getResult(0).getType();
    LoweringReshapeToShapeAssign(rewriter, op, new_type);
  } else {
    lowering_common<tpu::ReshapeOp>(rewriter, op, op->getResult(0).getType());
  }
}

} // namespace bm1684x
} // namespace tpu_mlir
