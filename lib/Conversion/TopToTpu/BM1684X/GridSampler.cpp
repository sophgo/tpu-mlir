//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "bmcpu_common.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void GridSamplerLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::GridSamplerOp op) const {
  int mode = op.getMode();
  if (mode == 0) {
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
                                                   op->getOperands(), attrs);
  } else {
    rewriter.replaceOpWithNewOp<tpu::GridSamplerOp>(op, op.getOutput().getType(),
                                                  op->getOperands(), op->getAttrs());
  }
}

void GridSamplerLowering::LoweringINT4(PatternRewriter &rewriter,
                                       top::GridSamplerOp op,
                                       bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void GridSamplerLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::GridSamplerOp op,
                                       bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void GridSamplerLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::GridSamplerOp op) const {
  LoweringF32(rewriter, op);
}

void GridSamplerLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::GridSamplerOp op) const {
  LoweringF32(rewriter, op);
}

void GridSamplerLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::GridSamplerOp op) const {
  llvm_unreachable("Not Implemented");
}
} // namespace bm1684x
} // namespace tpu_mlir
