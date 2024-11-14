//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-gridsampler"

namespace tpu_mlir {
namespace cv18xx {
void GridSamplerLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::GridSamplerOp op,
                                       bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

void GridSamplerLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::GridSamplerOp op) const {
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> cpu_param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("grid_sampler")));
  for (auto &attr : op->getAttrs()) {
    cpu_param.push_back(attr);
  }
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(cpu_param)));
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, op.getOutput().getType(),
                                                 op.getOperands(), attrs);
}
} // namespace cv18xx
} // namespace tpu_mlir
