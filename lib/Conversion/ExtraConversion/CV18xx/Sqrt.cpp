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
LogicalResult ConvertSqrtOp::matchAndRewrite(top::SqrtOp op,
                              PatternRewriter &rewriter) const{
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  Value input = op.getInput();
  operands.emplace_back(input);
  attrs.emplace_back(rewriter.getNamedAttr("exponent", rewriter.getF64FloatAttr(0.5)));
  rewriter.replaceOpWithNewOp<top::PowOp>(
      op, op.getOutput().getType().cast<RankedTensorType>(), operands, attrs);
  return success();
}
}
}
