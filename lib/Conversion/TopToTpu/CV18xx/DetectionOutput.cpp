//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-detection-output"

namespace tpu_mlir {
namespace cv18xx {

void loweringDetectionOutput(PatternRewriter &rewriter,
                             top::DetectionOutputOp op) {
  // lowering to cpu op
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("detectionoutput")));
  param.emplace_back(rewriter.getNamedAttr(
      "num_classes", rewriter.getI32IntegerAttr(op.getNumClasses())));
  param.emplace_back(rewriter.getNamedAttr(
      "background_label_id",
      rewriter.getI32IntegerAttr(op.getBackgroundLabelId())));
  param.emplace_back(rewriter.getNamedAttr(
      "nms_threshold",
      rewriter.getF32FloatAttr(op.getNmsThreshold().convertToDouble())));
  param.emplace_back(
      rewriter.getNamedAttr("top_k", rewriter.getI32IntegerAttr(op.getTopK())));
  param.emplace_back(rewriter.getNamedAttr(
      "keep_top_k", rewriter.getI32IntegerAttr(op.getKeepTopK())));
  param.emplace_back(rewriter.getNamedAttr(
      "confidence_threshold",
      rewriter.getF32FloatAttr(op.getConfidenceThreshold().convertToDouble())));
  param.emplace_back(rewriter.getNamedAttr(
      "code_type", rewriter.getStringAttr(op.getCodeType())));
  param.emplace_back(rewriter.getNamedAttr(
      "onnx_nms",
      rewriter.getI64IntegerAttr(op.getInputs().size() >= 3 ? 0 : 1)));
  param.emplace_back(rewriter.getNamedAttr(
      "share_location", rewriter.getBoolAttr(op.getShareLocation())));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands(op.getOperands().begin(), op.getOperands().end());
  mlir::Type new_type = getQuantFloatType(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_type, operands, attrs);
}

void DetectionOutputLowering::LoweringINT8(PatternRewriter &rewriter,
                                           top::DetectionOutputOp op,
                                           bool asymmetric) const {
  loweringDetectionOutput(rewriter, op);
}

void DetectionOutputLowering::LoweringBF16(PatternRewriter &rewriter,
                                           top::DetectionOutputOp op) const {
  loweringDetectionOutput(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
