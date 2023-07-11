//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void TopKLowering::LoweringF32(PatternRewriter &rewriter,
                               top::TopKOp op) const {
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> cpu_param;
  attrs.emplace_back(
      rewriter.getNamedAttr("cpu_op_name", rewriter.getStringAttr("topk")));

  for (auto &attr : op->getAttrs()) {
    cpu_param.push_back(attr);
  }

  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(cpu_param)));
  std::vector<Type> new_types;
  new_types.push_back(op.getValues().getType());
  if (!module::isNone(op.getIndices())) {
    auto shape = module::getShape(op.getIndices());
    auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
    new_types.push_back(new_type);
  } else {
    new_types.push_back(op.getIndices().getType());
  }
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_types, operands,
                                                 attrs);
}

void TopKLowering::LoweringINT8(PatternRewriter &rewriter, top::TopKOp op,
                                bool asymmetric) const {
  llvm_unreachable("Topk do not support Lowering to Int8");
}

} // namespace bm1684
} // namespace tpu_mlir
