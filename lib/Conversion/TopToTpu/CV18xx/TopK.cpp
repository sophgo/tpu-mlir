//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-tok"

namespace tpu_mlir {
namespace cv18xx {
void TopKLowering::LoweringINT8(PatternRewriter &rewriter, top::TopKOp op,
                                bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

void TopKLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::TopKOp op) const {
  if (op.getKT() || !module::isNone(op.getIndices())) {
    module::getModuleOp()->dump();
    llvm_unreachable("Not supported now");
  }

  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(
      rewriter.getNamedAttr("cpu_op_name", rewriter.getStringAttr("topk")));
  std::vector<NamedAttribute> cpu_op_param;
  cpu_op_param.emplace_back(
      rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(op.getAxis())));
  cpu_op_param.emplace_back(
      rewriter.getNamedAttr("K", rewriter.getI32IntegerAttr(op.getK())));
  cpu_op_param.emplace_back(
      rewriter.getNamedAttr("sorted", rewriter.getBoolAttr(op.getSorted())));
  cpu_op_param.emplace_back(
      rewriter.getNamedAttr("largest", rewriter.getBoolAttr(op.getLargest())));

  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(cpu_op_param)));
  std::vector<Value> operands(op.getOperands().begin(), op.getOperands().end());
  // auto newType = getQuantBF16Type(op.getOutput());
  std::vector<Type> types;
  types.push_back(op.getValues().getType());
  auto topk_op = rewriter.create<tpu::GenericCpuOp>(
      op.getValues().getLoc(), types, op->getOperands(), attrs);
  rewriter.replaceAllUsesWith(op.getValues(), topk_op.getOutputs());
  rewriter.eraseOp(op);
}
} // namespace cv18xx
} // namespace tpu_mlir
