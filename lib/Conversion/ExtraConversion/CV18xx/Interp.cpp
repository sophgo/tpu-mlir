//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertCV18XX.h"

namespace tpu_mlir {
namespace cv18xx {

LogicalResult ConvertInterpOp::matchAndRewrite(top::InterpOp op,
                                            PatternRewriter &rewriter) const {
  //implement
  std::string mode = op.getMode().str();
  double scale_h = op.getScaleH().convertToDouble();
  double scale_w = op.getScaleW().convertToDouble();
  if (mode == "nearest" && std::ceil(scale_h) == std::floor(scale_h)
      && std::ceil(scale_w) == std::floor(scale_w)) {
    llvm::errs()<<"Warning, if model is onnx format, it should be already converted in onnx_convert\n";
    //from torch
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getNamedAttr("scale_h", rewriter.getI64IntegerAttr((int64_t)scale_h)));
    attrs.emplace_back(rewriter.getNamedAttr("scale_w", rewriter.getI64IntegerAttr((int64_t)scale_w)));
    std::vector<Value> operands;
    operands.emplace_back(op.getInput());
    rewriter.replaceOpWithNewOp<top::UpsampleOp>(op, op.getType(), operands, attrs);
    return success();
  }
  return failure();
}

} // namespace cv18xx
} // namespace tpu_mlir
