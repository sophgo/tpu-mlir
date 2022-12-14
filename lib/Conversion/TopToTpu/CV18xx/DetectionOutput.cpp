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
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-detection-output"

namespace tpu_mlir {
namespace cv18xx {

void loweringDetectionOutput(PatternRewriter &rewriter,
                             top::DetectionOutputOp op) {
  auto o_shape = Module::getShape(op.output());
  // lowering to cpu op
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "operation_name", rewriter.getStringAttr("detectionoutput")));
  param.emplace_back(rewriter.getNamedAttr(
      "num_classes", rewriter.getI32IntegerAttr(op.num_classes())));
  param.emplace_back(rewriter.getNamedAttr(
      "background_label_id",
      rewriter.getI32IntegerAttr(op.background_label_id())));
  param.emplace_back(rewriter.getNamedAttr(
      "nms_threshold",
      rewriter.getF32FloatAttr(op.nms_threshold().convertToDouble())));
  param.emplace_back(
      rewriter.getNamedAttr("top_k", rewriter.getI32IntegerAttr(op.top_k())));
  param.emplace_back(rewriter.getNamedAttr(
      "keep_top_k", rewriter.getI32IntegerAttr(op.keep_top_k())));
  param.emplace_back(rewriter.getNamedAttr(
      "confidence_threshold",
      rewriter.getF32FloatAttr(op.confidence_threshold().convertToDouble())));
  param.emplace_back(rewriter.getNamedAttr(
      "code_type", rewriter.getStringAttr(op.code_type())));
  param.emplace_back(rewriter.getNamedAttr(
      "share_location", rewriter.getBoolAttr(op.share_location())));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands(op.getOperands().begin(), op.getOperands().end());
  mlir::Type new_type = getQuantFloatType(op.output());
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
