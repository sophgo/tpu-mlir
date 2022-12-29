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

#define DEBUG_TYPE "lowering-roipooling"
namespace tpu_mlir {
namespace cv18xx {
void loweringROIPooling(PatternRewriter &rewriter,
                             top::ROIPoolingOp op) {
  auto o_shape = module::getShape(op.output());
  // lowering to cpu op
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "operation_name", rewriter.getStringAttr("roi_pooling")));
  param.emplace_back(rewriter.getNamedAttr(
      "pooled_h",
      rewriter.getI32IntegerAttr(op.pooled_h())));
  param.emplace_back(rewriter.getNamedAttr(
      "pooled_w",
      rewriter.getI32IntegerAttr(op.pooled_w())));
  param.emplace_back(rewriter.getNamedAttr(
      "spatial_scale",
      rewriter.getF32FloatAttr(op.spatial_scale().convertToDouble())));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands(op.getOperands().begin(), op.getOperands().end());
  mlir::Type new_type = getQuantFloatType(op.output());
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
}
}
