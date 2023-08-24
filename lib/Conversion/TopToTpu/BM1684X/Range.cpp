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

void RangeTryLowering::Lowering(PatternRewriter &rewriter,
                                top::RangeOp op) const {
  auto prev_op = op.getLimit().getDefiningOp();
  if (!prev_op->hasTrait<trait::ShapeProducer>()) {
    return;
  }

  std::vector<Value> operands;
  auto i32_t_type =
      RankedTensorType::get({1}, rewriter.getIntegerType(32, true));
  if (module::isNone(op.getStart())) {
    std::vector<int32_t> start_v{0};
    auto start_val = top::WeightOp::create(op, "i32", start_v, i32_t_type);
    auto hdOp = insert_device2host(start_val, start_val.getType());
    operands.push_back(hdOp);
  } else if (auto start_w = dyn_cast<top::WeightOp>(op.getStart().getDefiningOp())) {
    auto intOp = start_w.clone_int(op);
    auto hdOp = insert_device2host(intOp, intOp.getType());
    operands.push_back(hdOp);
  } else {
    operands.push_back(op.getStart());
  }
  operands.push_back(op.getLimit());
  if (module::isNone(op.getStart())) {
    std::vector<int32_t> delta_v{0};
    auto delta_val = top::WeightOp::create(op, "i32", delta_v, i32_t_type);
    auto hdOp = insert_device2host(delta_val, delta_val.getType());
    operands.push_back(hdOp);
  } else if (auto delta_w = dyn_cast<top::WeightOp>(op.getDelta().getDefiningOp())) {
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

void RangeLowering::LoweringF32(PatternRewriter &rewriter,
                                top::RangeOp op) const {
  llvm_unreachable("Not implemented");
}
void RangeLowering::LoweringINT4(PatternRewriter &rewriter, top::RangeOp op,
                                 bool asymmetric) const {
  llvm_unreachable("Not implemented");
}
void RangeLowering::LoweringINT8(PatternRewriter &rewriter, top::RangeOp op,
                                 bool asymmetric) const {
  llvm_unreachable("Not implemented");
}

void RangeLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::RangeOp op) const {
  llvm_unreachable("Not implemented");
}

void RangeLowering::LoweringF16(PatternRewriter &rewriter,
                                top::RangeOp op) const {
  llvm_unreachable("Not implemented");
}

void RangeLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::RangeOp op) const {
  llvm_unreachable("Not implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
