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

static void _try_insert_device2host(top::SliceOp op) {
  const bool is_dynamic = !module::isNone(op.getOffsetT()) ||
                          !module::isNone(op.getEndsT()) ||
                          !module::isNone(op.getStepsT());
  if (is_dynamic) {
    for (int idx = 1; idx < 4; ++idx) {
      if (!module::isNone(op->getOperand(idx)))
        try_insert_device2host(op.getOperation(), idx);
    }
  }
}

void SliceTryLowering::Lowering(PatternRewriter &rewriter,
                                top::SliceOp op) const {
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
  rewriter.replaceOpWithNewOp<tpu::ShapeSliceOp>(op, new_type, op.getOperands(),
                                                 attrs);
}

void SliceLowering::LoweringF32(PatternRewriter &rewriter,
                                top::SliceOp op) const {
  _try_insert_device2host(op);
  auto input = op.getOperand(0);
  auto stype = module::getStorageType(input);

  if (stype.isInteger(32)) {
    std::vector<NamedAttribute> attrs;
    for (auto &attr : op->getAttrs()) {
      attrs.push_back(attr);
    }
    std::vector<Value> operands;
    for (auto opd : op->getOperands()) {
      operands.push_back(opd);
    }
    auto noneOp = module::getNoneOp(op);
    operands.push_back(noneOp); // buffer
    rewriter.replaceOpWithNewOp<tpu::SliceOp>(op, stype, operands, attrs);
    return;
  }

  lowering_common_f32<tpu::SliceOp>(rewriter, op, 5);
}
void SliceLowering::LoweringINT4(PatternRewriter &rewriter, top::SliceOp op,
                                 bool asymmetric) const {
  _try_insert_device2host(op);
  LoweringINT8(rewriter, op, asymmetric);
}
void SliceLowering::LoweringINT8(PatternRewriter &rewriter, top::SliceOp op,
                                 bool asymmetric) const {
  _try_insert_device2host(op);
  lowering_common_int8<tpu::SliceOp>(rewriter, op, asymmetric, 5);
}

void SliceLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::SliceOp op) const {
  _try_insert_device2host(op);
  lowering_common_bf16<tpu::SliceOp>(rewriter, op, 5);
}

void SliceLowering::LoweringF16(PatternRewriter &rewriter,
                                top::SliceOp op) const {
  _try_insert_device2host(op);
  lowering_common_f16<tpu::SliceOp>(rewriter, op, 5);
}

void SliceLowering::LoweringF8(PatternRewriter &rewriter,
                               top::SliceOp op) const {
  _try_insert_device2host(op);
  bool isE4 = module::getMode() == module::Mode::F8E4M3;
  lowering_common_f8<tpu::SliceOp>(rewriter, op, isE4, 5);
}

void SliceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::SliceOp op) const {
  _try_insert_device2host(op);
  lowering_common<tpu::SliceOp>(rewriter, op, op.getOutput().getType(), 5);
}

} // namespace bm1684x
} // namespace tpu_mlir
