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

#define DEBUG_TYPE "lowering-sub"

namespace tpu_mlir {
namespace cv18xx {
static int is_bcast(top::SubOp op) {
  std::vector<int64_t> shape0;
  std::vector<int64_t> shape1;
  int bcast = 0;
  module::getShapeVec(op.inputs()[0], shape0);
  module::getShapeVec(op.inputs()[1], shape1);
  auto prod0 = std::accumulate(shape0.begin(), shape0.end(), 1,
                               std::multiplies<int64_t>());
  auto prod1 = std::accumulate(shape1.begin(), shape1.end(), 1,
                               std::multiplies<int64_t>());
  auto sub = prod0 - prod1;
  if (shape0.size() == shape1.size() && sub == 0) {
    for (int i = 0; i < shape0.size(); i++) {
      if (shape0[i] != shape1[i]) {
        bcast = 1;
        break;
      }
    }
  }
  if (bcast) {
    auto len = std::min(shape0.size(), shape1.size());
    for (int i = 0; i < len; i++) {
      int dim_a = shape0[shape0.size() - 1 - i];
      int dim_b = shape1[shape1.size() - 1 - i];
      if (dim_b != dim_b &&
          ((sub > 0 && dim_b != 1) || (sub < 0 && dim_a != 1))) {
        llvm_unreachable("Only broadcast right operand supported.");
      }
    }
  }
  return bcast;
}

void SubLowering::LoweringINT8(PatternRewriter &rewriter, top::SubOp op,
                               bool asymmetric) const {
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto v : op->getOperands()) {
    if (isa<top::WeightOp>(v.getDefiningOp())) {
      LoweringBF16(rewriter, op);
      return;
    }
  }
  std::vector<int64_t> rshift_v(1);
  std::vector<int64_t> multiplier_v(nInputs, 1);
  std::vector<float> qscale(nInputs, 1.0);
  float max_qscale = 0.0;
  assert(nInputs == 2);
  auto bcast = is_bcast(op);
  auto coeff_v = module::getF64Array(op.coeff(), 2, 1.0);
  assert(coeff_v->at(0) == 1 && coeff_v->at(1) == 1);
  if (!bcast) {
    coeff_v->at(1) = -1;
  }
  double o_scale = module::getThreshold(op.output());
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    operands.push_back(input);
    double i_scale = module::getThreshold(input);
    auto scale_f = i_scale / o_scale;
    qscale[i] = coeff_v->at(i) * scale_f;
  }

  for (auto &q : qscale) {
    if (max_qscale < std::abs(q)) {
      max_qscale = std::abs(q);
    }
  }
  int64_t multiplier = 0;
  int64_t shift = 0;
  getRShiftAndMultiplierFromQScale(max_qscale, &multiplier, &shift, false);

  rshift_v[0] = shift;
  for (int i = 0; i < nInputs; ++i) {
    multiplier_v[i] = getMultiplierI8FromQScaleAndRShift(qscale[i], shift);
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.do_reluAttr()));
  attrs.push_back(rewriter.getNamedAttr("relu_limit", op.relu_limitAttr()));
  attrs.push_back(rewriter.getNamedAttr("coeff", rewriter.getF64ArrayAttr({1, 1})));
  attrs.push_back(rewriter.getNamedAttr(
      "multipliers", rewriter.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(rshift_v)));
  auto newType = getQuantInt8Type(op.output());
  if (!bcast) {
    rewriter.replaceOpWithNewOp<tpu::AddOp>(op.getOperation(), newType,
                                            operands, attrs);
  } else {
    // todo  if prod(shape0) < prod(shape1) result mul -1 here
    attrs.push_back(rewriter.getNamedAttr("is_reverse", op.is_reverseAttr()));
    rewriter.replaceOpWithNewOp<tpu::SubOp>(op.getOperation(), newType,
                                            operands, attrs);
  }
  return;
}

void SubLowering::LoweringBF16(PatternRewriter &rewriter, top::SubOp op) const {
  if (!is_bcast(op)) {
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("do_relu", op.do_reluAttr()));
    attrs.push_back(rewriter.getNamedAttr("relu_limit", op.relu_limitAttr()));
    attrs.push_back(
        rewriter.getNamedAttr("coeff", rewriter.getF64ArrayAttr({1., -1.})));
    auto newType = getQuantBF16Type(op.output());
    rewriter.replaceOpWithNewOp<tpu::AddOp>(op, newType, op->getOperands(),
                                            attrs);
  } else {
    lowering_common_bf16<tpu::SubOp>(rewriter, op);
  }
}

} // namespace cv18xx
} // namespace tpu_mlir
