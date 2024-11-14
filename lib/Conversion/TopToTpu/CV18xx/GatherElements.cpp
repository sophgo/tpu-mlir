
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-gatherelements"

namespace tpu_mlir {
namespace cv18xx {
void GatherElementsLowering::LoweringINT8(PatternRewriter &rewriter,
                                          top::GatherElementsOp op,
                                          bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

void GatherElementsLowering::LoweringBF16(PatternRewriter &rewriter,
                                          top::GatherElementsOp op) const {
  // std::vector<Value> operands;
  // std::vector<NamedAttribute> attrs;
  // std::vector<NamedAttribute> cpu_op_param;
  // attrs.emplace_back(rewriter.getNamedAttr("cpu_op_name",
  // rewriter.getStringAttr("gatherelements_pt")));
  // cpu_op_param.emplace_back(rewriter.getNamedAttr("axis",
  // rewriter.getI32IntegerAttr(op.getAxis())));
  // attrs.emplace_back(rewriter.getNamedAttr("param",
  // rewriter.getDictionaryAttr(cpu_op_param))); if (auto indiceOp =
  // dyn_cast_or_null<top::WeightOp>(op.getIndices().getDefiningOp())) {
  //   operands.emplace_back(indiceOp.clone_int(op));
  // } else {
  //   operands.emplace_back(op.getIndices());
  // }
  // if (auto inputOp =
  // dyn_cast_or_null<top::WeightOp>(op.getInput().getDefiningOp())) {
  //   operands.emplace_back(inputOp.clone_bf16(op));
  // } else {
  //   operands.emplace_back(op.getInput());
  // }
  // auto newType = getQuantBF16Type(op.getOutput());
  // rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op.getOperation(), newType,
  //                                            operands, attrs);
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> cpu_param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("gatherelements_pt")));
  for (auto &attr : op->getAttrs()) {
    cpu_param.push_back(attr);
  }
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(cpu_param)));
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
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, op.getOutput().getType(),
                                                 operands, attrs);
}
} // namespace cv18xx
} // namespace tpu_mlir
