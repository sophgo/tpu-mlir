//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-InstanceNorm"

namespace tpu_mlir {
namespace cv18xx {

void loweringInstanceNorm(PatternRewriter &rewriter, top::InstanceNormOp op) {
  // lowering to cpu op
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("instance_norm")));
  param.emplace_back(rewriter.getNamedAttr(
      "variance_epsilon",
      rewriter.getF32FloatAttr(op.getEps().convertToDouble())));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands;
  operands.emplace_back(op.getInput());
  Value weight = op.getWeight();
  if (isa<top::NoneOp>(op.getWeight().getDefiningOp())) {
    auto shape = module::getShape(op.getInput());
    assert(shape.size() > 1);
    auto weight_type = RankedTensorType::get({shape[1]}, rewriter.getF32Type());
    weight = top::WeightOp::create(
        op, module::getName(op.getOutput()).str() + "_weight",
        std::vector<float>(shape[1], 1), weight_type);
  }
  operands.emplace_back(weight);
  Value bias = op.getBias();
  if (isa<top::NoneOp>(op.getBias().getDefiningOp())) {
    auto shape = module::getShape(op.getInput());
    assert(shape.size() > 1);
    auto bias_type = RankedTensorType::get({shape[1]}, rewriter.getF32Type());
    bias = top::WeightOp::create(
        op, module::getName(op.getOutput()).str() + "_bias",
        std::vector<float>(shape[1], 0), bias_type);
  }
  operands.emplace_back(bias);
  mlir::Type new_type = getQuantFloatType(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_type, operands, attrs);
}

void InstanceNormLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::InstanceNormOp op,
                                        bool asymmetric) const {
  loweringInstanceNorm(rewriter, op);
}

void InstanceNormLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::InstanceNormOp op) const {
  loweringInstanceNorm(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
