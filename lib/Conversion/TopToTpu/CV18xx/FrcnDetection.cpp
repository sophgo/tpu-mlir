//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-frcn-detection"
namespace tpu_mlir {
namespace cv18xx {
void loweringFrcnDetection(PatternRewriter &rewriter, top::FrcnDetectionOp op) {
  // lowering to cpu op
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("frcn_detection")));
  param.emplace_back(rewriter.getNamedAttr(
      "class_num", rewriter.getI32IntegerAttr(op.getClassNum())));
  param.emplace_back(rewriter.getNamedAttr(
      "keep_topk", rewriter.getI32IntegerAttr(op.getKeepTopk())));
  param.emplace_back(rewriter.getNamedAttr(
      "obj_threshold",
      rewriter.getF32FloatAttr(op.getObjThreshold().convertToDouble())));
  param.emplace_back(rewriter.getNamedAttr(
      "nms_threshold",
      rewriter.getF32FloatAttr(op.getNmsThreshold().convertToDouble())));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands(op.getOperands().begin(), op.getOperands().end());
  mlir::Type new_type = getQuantFloatType(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_type, operands, attrs);
}

void FrcnDetectionLowering::LoweringINT8(PatternRewriter &rewriter,
                                         top::FrcnDetectionOp op,
                                         bool asymmetric) const {
  loweringFrcnDetection(rewriter, op);
}

void FrcnDetectionLowering::LoweringBF16(PatternRewriter &rewriter,
                                         top::FrcnDetectionOp op) const {
  loweringFrcnDetection(rewriter, op);
}
} // namespace cv18xx
} // namespace tpu_mlir
