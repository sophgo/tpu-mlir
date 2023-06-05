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

void ScatterNDLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::ScatterNDOp op) const {
  std::vector<Value> operands;
  operands.push_back(op.getInputData());
  auto indices_op = dyn_cast<top::WeightOp>(op.getIndices().getDefiningOp());
  if (indices_op) {
    // convert fp32 indices into int32
    auto indices_data = indices_op.read<float>();
    std::vector<int32_t> indices_int32_v(
        module::getNumElements(op.getIndices()));
    for (int i = 0; i < module::getNumElements(op.getIndices()); ++i) {
      indices_int32_v[i] = static_cast<int32_t>(indices_data->at(i));
    }
    auto new_type = RankedTensorType::get(module::getShape(op.getIndices()),
                                          rewriter.getI32Type());
    i32_array_t indices_int32 =
        std::make_shared<std::vector<int32_t>>(indices_int32_v);
    auto new_indices_op =
        top::WeightOp::create(op, "indices_int32", *indices_int32, new_type);
    operands.push_back(new_indices_op);
  } else {
    operands.push_back(op.getIndices());
  }

  operands.push_back(op.getUpdates());

  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);

  rewriter.replaceOpWithNewOp<tpu::ScatterNDOp>(op, op.getOutput().getType(),
                                                operands, op->getAttrs());
}

void ScatterNDLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::ScatterNDOp op,
                                     bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void ScatterNDLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::ScatterNDOp op,
                                     bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void ScatterNDLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::ScatterNDOp op) const {
  LoweringF32(rewriter, op);
}

void ScatterNDLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::ScatterNDOp op) const {
  LoweringF32(rewriter, op);
}

void ScatterNDLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::ScatterNDOp op) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir
