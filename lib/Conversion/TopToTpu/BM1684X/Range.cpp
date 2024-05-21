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

static void _try_insert_device2host(top::RangeOp op) {
  const bool is_dynamic = !module::isNone(op.getLimit());
  if (is_dynamic) {
    try_insert_device2host(op.getOperation(), 1);
  }
}

void RangeLowering::LoweringF32(PatternRewriter &rewriter,
                                top::RangeOp op) const {
  _try_insert_device2host(op);

  std::vector<Value> operands;
  auto i32_t_type =
      RankedTensorType::get({1}, rewriter.getIntegerType(32, true));
  if (module::isNone(op.getStart())) {
    std::vector<int32_t> start_v{0};
    auto start_val = top::WeightOp::create(op, "i32", start_v, i32_t_type);
    auto hdOp = insert_device2host(start_val, start_val.getType());
    operands.push_back(hdOp);
  } else if (auto start_w =
                 dyn_cast<top::WeightOp>(op.getStart().getDefiningOp())) {
    auto intOp = start_w.clone_int(op);
    auto hdOp = insert_device2host(intOp, intOp.getType());
    operands.push_back(hdOp);
  } else {
    operands.push_back(op.getStart());
  }
  operands.push_back(op.getLimit());
  if (module::isNone(op.getDelta())) {
    std::vector<int32_t> delta_v{0};
    auto delta_val = top::WeightOp::create(op, "i32", delta_v, i32_t_type);
    auto hdOp = insert_device2host(delta_val, delta_val.getType());
    operands.push_back(hdOp);
  } else if (auto delta_w =
                 dyn_cast<top::WeightOp>(op.getDelta().getDefiningOp())) {
    auto intOp = delta_w.clone_int(op);
    auto hdOp = insert_device2host(intOp, intOp.getType());
    operands.push_back(hdOp);
  } else {
    operands.push_back(op.getDelta());
  }

  std::vector<NamedAttribute> attrs;
  Type new_type = RankedTensorType::get(module::getShape(op.getOutput()),
                                        Float32Type::get(op.getContext()));
  rewriter.replaceOpWithNewOp<tpu::RangeOp>(op, new_type, operands, attrs);
}
void RangeLowering::LoweringINT4(PatternRewriter &rewriter, top::RangeOp op,
                                 bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void RangeLowering::LoweringINT8(PatternRewriter &rewriter, top::RangeOp op,
                                 bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void RangeLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::RangeOp op) const {
  LoweringF32(rewriter, op);
}

void RangeLowering::LoweringF16(PatternRewriter &rewriter,
                                top::RangeOp op) const {
  LoweringF32(rewriter, op);
}

void RangeLowering::LoweringF8(PatternRewriter &rewriter,
                               top::RangeOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RangeLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::RangeOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
