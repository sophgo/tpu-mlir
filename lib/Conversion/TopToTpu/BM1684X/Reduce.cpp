//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Support/Float8.h"
namespace tpu_mlir {
namespace bm1684x {

void ReduceTryLowering::Lowering(PatternRewriter &rewriter,
                                 top::ReduceOp op) const {
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
  rewriter.replaceOpWithNewOp<tpu::ShapeReduceOp>(op, new_type, op.getOperand(),
                                                  attrs);
}

void ReduceLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::ReduceOp op) const {
  lowering_common_f32<tpu::ReduceOp>(rewriter, op, 3);
}
void ReduceLowering::LoweringINT4(PatternRewriter &rewriter, top::ReduceOp op,
                                  bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ReduceLowering::LoweringINT8(PatternRewriter &rewriter, top::ReduceOp op,
                                  bool asymmetric) const {
  if (!module::isMARS3() && !module::isSGTPUV8()) {
    LoweringF16(rewriter, op);
  } else {
    LoweringBF16(rewriter, op);
  }
}

void ReduceLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::ReduceOp op) const {
  lowering_common_bf16<tpu::ReduceOp>(rewriter, op, 3);
}

void ReduceLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::ReduceOp op) const {
  lowering_common_f16<tpu::ReduceOp>(rewriter, op, 3);
}

void ReduceLowering::LoweringF8(PatternRewriter &rewriter,
                                top::ReduceOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  lowering_common_f16<tpu::ReduceOp>(rewriter, op, 3);
}

void ReduceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::ReduceOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
