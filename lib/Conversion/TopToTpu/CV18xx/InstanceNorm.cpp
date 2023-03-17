//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-InstanceNorm"

namespace tpu_mlir {
namespace cv18xx {

void loweringInstanceNorm(PatternRewriter &rewriter, top::InstanceNormOp op) {
  auto o_shape = module::getShape(op.getOutput());
  // lowering to cpu op
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("instance_norm")));
  param.emplace_back(rewriter.getNamedAttr(
      "variance_epsilon",
      rewriter.getF32FloatAttr(op.getEps().convertToDouble())));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands(op.getOperands().begin(), op.getOperands().end());
  mlir::Type new_type = getQuantFloatType(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_type, operands, attrs);
}

void InstanceNormLowering::LoweringINT8(PatternRewriter &rewriter,
                                         top::InstanceNormOp op,
                                         bool asymmetric) const {
  loweringInstanceNorm(rewriter, op);
}

void InstanceNormLowering::LoweringBF16(PatternRewriter &rewriter,
                                         top::InstanceNormOp op) const {
  loweringInstanceNorm(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
