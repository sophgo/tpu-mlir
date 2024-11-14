//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-retinaface-detection"
namespace tpu_mlir {
namespace cv18xx {
void loweringRetinaFaceDetection(PatternRewriter &rewriter,
                                 top::RetinaFaceDetectionOp op) {
  // lowering to cpu op
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("retinaface_detection")));
  param.emplace_back(rewriter.getNamedAttr(
      "keep_topk", rewriter.getI64IntegerAttr(op.getKeepTopk())));
  param.emplace_back(rewriter.getNamedAttr(
      "confidence_threshold",
      rewriter.getF64FloatAttr(op.getConfidenceThreshold().convertToDouble())));
  param.emplace_back(rewriter.getNamedAttr(
      "nms_threshold",
      rewriter.getF64FloatAttr(op.getNmsThreshold().convertToDouble())));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands(op.getOperands().begin(), op.getOperands().end());
  mlir::Type new_type = getQuantFloatType(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_type, operands, attrs);
}

void RetinaFaceDetectionLowering::LoweringINT8(PatternRewriter &rewriter,
                                               top::RetinaFaceDetectionOp op,
                                               bool asymmetric) const {
  loweringRetinaFaceDetection(rewriter, op);
}

void RetinaFaceDetectionLowering::LoweringBF16(
    PatternRewriter &rewriter, top::RetinaFaceDetectionOp op) const {
  loweringRetinaFaceDetection(rewriter, op);
}
} // namespace cv18xx
} // namespace tpu_mlir
