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

void loweringYoloDetection(PatternRewriter &rewriter,
                             top::YoloDetectionOp op) {
  auto o_shape = module::getShape(op.output());
  // lowering to cpu op
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "operation_name", rewriter.getStringAttr("yolo_detection")));
  param.emplace_back(rewriter.getNamedAttr(
      "class_num", rewriter.getI32IntegerAttr(op.class_num())));
  param.emplace_back(rewriter.getNamedAttr(
      "net_input_h",
      rewriter.getI32IntegerAttr(op.net_input_h())));
  param.emplace_back(rewriter.getNamedAttr(
      "net_input_w",
      rewriter.getI32IntegerAttr(op.net_input_w())));
  param.emplace_back(rewriter.getNamedAttr(
      "nms_threshold",
      rewriter.getF32FloatAttr(op.nms_threshold().convertToDouble())));
  param.emplace_back(rewriter.getNamedAttr(
      "obj_threshold",
      rewriter.getF32FloatAttr(op.obj_threshold().convertToDouble())));
  param.emplace_back(rewriter.getNamedAttr(
      "keep_topk", rewriter.getI32IntegerAttr(op.keep_topk())));
  param.emplace_back(rewriter.getNamedAttr(
      "spp_net", rewriter.getBoolAttr(op.spp_net())));
  param.emplace_back(rewriter.getNamedAttr(
      "tiny", rewriter.getBoolAttr(op.tiny())));
  param.emplace_back(rewriter.getNamedAttr(
      "yolo_v4", rewriter.getBoolAttr(op.yolo_v4())));
  param.emplace_back(rewriter.getNamedAttr(
      "anchors", rewriter.getStringAttr(op.anchors())));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands(op.getOperands().begin(), op.getOperands().end());
  mlir::Type new_type = getQuantFloatType(op.output());
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_type, operands, attrs);
}

void YoloDetectionLowering::LoweringINT8(PatternRewriter &rewriter,
                                           top::YoloDetectionOp op,
                                           bool asymmetric) const {
  loweringYoloDetection(rewriter, op);
}

void YoloDetectionLowering::LoweringBF16(PatternRewriter &rewriter,
                                           top::YoloDetectionOp op) const {
  loweringYoloDetection(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
