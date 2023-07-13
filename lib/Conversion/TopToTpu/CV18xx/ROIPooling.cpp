//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-roipooling"
namespace tpu_mlir {
namespace cv18xx {
void loweringROIPooling(PatternRewriter &rewriter, top::ROIPoolingOp op) {
  // lowering to cpu op
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("roi_pooling")));
  param.emplace_back(rewriter.getNamedAttr(
      "pooled_h", rewriter.getI32IntegerAttr(op.getPooledH())));
  param.emplace_back(rewriter.getNamedAttr(
      "pooled_w", rewriter.getI32IntegerAttr(op.getPooledW())));
  param.emplace_back(rewriter.getNamedAttr(
      "spatial_scale",
      rewriter.getF32FloatAttr(op.getSpatialScale().convertToDouble())));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands(op.getOperands().begin(), op.getOperands().end());
  mlir::Type new_type = getQuantFloatType(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_type, operands, attrs);
}

void ROIPoolingLowering::LoweringINT8(PatternRewriter &rewriter,
                                      top::ROIPoolingOp op,
                                      bool asymmetric) const {
  loweringROIPooling(rewriter, op);
}

void ROIPoolingLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::ROIPoolingOp op) const {
  loweringROIPooling(rewriter, op);
}
} // namespace cv18xx
} // namespace tpu_mlir
