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

static void GatherND_lowering_common(PatternRewriter &rewriter,
                                     top::GatherNDOp op) {

  std::vector<Value> operands;
  operands.push_back(op.getInput());

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

  auto batch_dims = op.getBatchDims();
  if (batch_dims == 0) {
    rewriter.replaceOpWithNewOp<tpu::GatherNDOp>(op, op.getOutput().getType(),
                                                 operands, op->getAttrs());
  } else {
    // TODO implement batch_dims backend
    std::vector<NamedAttribute> attrs;
    std::vector<NamedAttribute> cpu_param;
    attrs.emplace_back(rewriter.getNamedAttr(
        "cpu_op_name", rewriter.getStringAttr("gathernd_tf")));
    for (auto &attr : op->getAttrs()) {
      cpu_param.push_back(attr);
    }
    attrs.emplace_back(
        rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(cpu_param)));

    rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, op.getOutput().getType(),
                                                   operands, attrs);
  }
}

void GatherNDLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::GatherNDOp op) const {
  GatherND_lowering_common(rewriter, op);
}

void GatherNDLowering::LoweringINT4(PatternRewriter &rewriter,
                                    top::GatherNDOp op, bool asymmetric) const {
  GatherND_lowering_common(rewriter, op);
}

void GatherNDLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::GatherNDOp op, bool asymmetric) const {
  if (module::isWeight(op.getInput())) {
    LoweringF32(rewriter, op);
    return;
  }
  GatherND_lowering_common(rewriter, op);
}

void GatherNDLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::GatherNDOp op) const {
  GatherND_lowering_common(rewriter, op);
}

void GatherNDLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::GatherNDOp op) const {
  GatherND_lowering_common(rewriter, op);
}

void GatherNDLowering::LoweringF8(PatternRewriter &rewriter,
                                  top::GatherNDOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void GatherNDLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::GatherNDOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}
} // namespace bm1684x
} // namespace tpu_mlir
