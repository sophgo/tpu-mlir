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

void loweringYoloDetection(PatternRewriter &rewriter, top::YoloDetectionOp op) {
  // lowering to cpu op
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("yolo_detection")));
  param.emplace_back(rewriter.getNamedAttr(
      "class_num", rewriter.getI32IntegerAttr(op.getClassNum())));
  param.emplace_back(rewriter.getNamedAttr(
      "net_input_h", rewriter.getI32IntegerAttr(op.getNetInputH())));
  param.emplace_back(rewriter.getNamedAttr(
      "net_input_w", rewriter.getI32IntegerAttr(op.getNetInputW())));
  param.emplace_back(rewriter.getNamedAttr(
      "nms_threshold",
      rewriter.getF32FloatAttr(op.getNmsThreshold().convertToDouble())));
  param.emplace_back(rewriter.getNamedAttr(
      "obj_threshold",
      rewriter.getF32FloatAttr(op.getObjThreshold().convertToDouble())));
  param.emplace_back(rewriter.getNamedAttr(
      "keep_topk", rewriter.getI32IntegerAttr(op.getKeepTopk())));
  auto version = op.getVersion();
  bool spp_net = false, tiny = false, yolov4 = false;
  if (version == "yolov4") {
    yolov4 = true;
  } else if (version == "yolov3_spp") {
    spp_net = true;
  } else if (version == "yolov3_tiny") {
    tiny = true;
  }
  param.emplace_back(
      rewriter.getNamedAttr("spp_net", rewriter.getBoolAttr(spp_net)));
  param.emplace_back(rewriter.getNamedAttr("tiny", rewriter.getBoolAttr(tiny)));
  param.emplace_back(
      rewriter.getNamedAttr("yolo_v4", rewriter.getBoolAttr(yolov4)));
  auto anchors = module::getF64Array(op.getAnchors());
  auto num_anchor = anchors->size();
  std::stringstream ss;
  for (int i = 0; i < num_anchor; i++) {
    ss << anchors->at(i);
    if (i < num_anchor - 1) {
      ss << ",";
    }
  }
  param.emplace_back(
      rewriter.getNamedAttr("anchors", rewriter.getStringAttr(ss.str())));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands(op.getOperands().begin(), op.getOperands().end());
  mlir::Type new_type = getQuantFloatType(op.getOutput());
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
