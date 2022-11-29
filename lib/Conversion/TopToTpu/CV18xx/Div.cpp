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

#define DEBUG_TYPE "lowering-div"

namespace tpu_mlir {
namespace cv18xx {

void DivLowering::LoweringINT8(PatternRewriter &rewriter, top::DivOp divOp,
                               bool asymmetric) const {
  std::vector<Value> operands;
  auto input_shape1 = Module::getShape(divOp.inputs()[0]);
  auto input_shape2 = Module::getShape(divOp.inputs()[1]);

  auto weight_op = dyn_cast<top::WeightOp>(divOp.inputs()[1].getDefiningOp());
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(rewriter.getNamedAttr("do_relu", divOp.do_reluAttr()));
  attrs.emplace_back(
      rewriter.getNamedAttr("relu_limit", divOp.relu_limitAttr()));
  operands.emplace_back(divOp.inputs()[0]);
  if (weight_op) {
    assert(weight_op);
    auto const_f32 = weight_op.read<float>();
    for (auto &const_value : *const_f32) {
      const_value = 1 / const_value;
    }
    auto weight_type = weight_op.getType().cast<RankedTensorType>();
    auto new_weight_operand =
        top::WeightOp::create(divOp, "weight", *const_f32, weight_type);
    operands.emplace_back(new_weight_operand);
    rewriter.replaceOpWithNewOp<top::MulConstOp>(
        divOp.getOperation(), divOp.output().getType().cast<RankedTensorType>(),
        operands, attrs);
    return;
  } else {
    rewriter.setInsertionPointAfterValue(divOp.inputs()[1]);
    std::string name = Module::getName(divOp.inputs()[1]).str() + "_reciprocal";
    auto loc = NameLoc::get(rewriter.getStringAttr(name));
    std::vector<NamedAttribute> reci_attrs;
    reci_attrs.emplace_back(rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(1.0)));
    auto reciprocal_op = rewriter.create<top::ReciprocalOp>(
        loc, divOp.inputs()[1].getType().cast<RankedTensorType>(),
        ValueRange{divOp.inputs()[1]}, reci_attrs);
    operands.emplace_back(reciprocal_op.output());
    rewriter.replaceOpWithNewOp<top::MulOp>(
        divOp.getOperation(), divOp.output().getType().cast<RankedTensorType>(),
        operands, attrs);
    return;
  }
}

void DivLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::DivOp divOp) const {
  lowering_common_bf16<tpu::DivOp>(rewriter, divOp.getOperation());
}

} // namespace cv18xx
} // namespace tpu_mlir
