//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void ConstantFillTryLowering::Lowering(PatternRewriter &rewriter,
                                       top::ConstantFillOp op) const {
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  Type new_type = getQuantFloatType<Float32Type>(op->getResult(0));
  rewriter.replaceOpWithNewOp<tpu::ConstantFillOp>(op, new_type, op.getInput(),
                                                   attrs);
}

} // namespace bm1684
} // namespace tpu_mlir