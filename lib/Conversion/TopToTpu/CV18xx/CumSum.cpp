//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-cumsum"

namespace tpu_mlir {
namespace cv18xx {
void CumSumLowering::LoweringINT8(PatternRewriter &rewriter, top::CumSumOp op,
                                  bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

void CumSumLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::CumSumOp op) const {
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(
      rewriter.getNamedAttr("cpu_op_name", rewriter.getStringAttr("cumsum")));
  std::vector<NamedAttribute> cpu_op_param;
  cpu_op_param.emplace_back(
      rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(op.getAxis())));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(cpu_op_param)));
  std::vector<Value> operands(op.getOperands().begin(), op.getOperands().end());
  // auto newType = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, op.getOutput().getType(),
                                                 operands, attrs);
}
} // namespace cv18xx
} // namespace tpu_mlir
