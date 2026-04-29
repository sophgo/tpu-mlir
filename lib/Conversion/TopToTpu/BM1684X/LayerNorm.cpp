//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

static void LoweringLayerNorm(PatternRewriter &rewriter, top::LayerNormOp op,
                              Type type) {
  rewriter.setInsertionPointAfter(op);
  auto clone_weight_by_type = [&](Value opd) -> Value {
    if (!module::isWeight(opd)) {
      return opd;
    }
    auto weightOp = opd.getDefiningOp<top::WeightOp>();
    if (type.isBF16()) {
      return weightOp.clone_bf16(op);
    } else if (type.isF16()) {
      return weightOp.clone_f16(op);
    }
    return opd;
  };
  auto input = op->getOperand(0);
  auto weight = clone_weight_by_type(op->getOperand(1));
  auto bias = clone_weight_by_type(op->getOperand(2));
  auto input_shape = module::getShape(op.getInput());
  int64_t inner_num = 1;
  for (int i = op.getAxis(); i < input_shape.size(); ++i) {
    inner_num *= input_shape[i];
  }
  const bool have_weight = !op.getWeight().getType().isa<NoneType>();
  const bool have_bias = !op.getBias().getType().isa<NoneType>();
  const bool bias_only = have_bias && !have_weight;
  const bool weight_mismatch =
      have_weight && module::getNumElements(op.getWeight()) != inner_num;
  const bool bias_mismatch =
      have_bias && module::getNumElements(op.getBias()) != inner_num;
  // For BM1684X F16, bias-only LayerNorm can be unstable in kernel affine path.
  // Always rewrite it to LayerNorm(no affine) + Add(bias).
  const bool need_split_affine = weight_mismatch || bias_mismatch || bias_only;
  std::vector<Value> opds;
  opds.reserve(5);
  auto none = module::getNoneOp(op);
  opds.push_back(input);
  opds.push_back(need_split_affine ? none : weight);
  opds.push_back(need_split_affine ? none : bias);
  opds.push_back(none);
  opds.push_back(none);

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  std::vector<Type> new_types;
  new_types.reserve(3);
  auto out = op.getResult();
  if (type.isF16()) {
    new_types.push_back(getQuantF16Type(out));
  } else if (type.isBF16()) {
    new_types.push_back(getQuantBF16Type(out));
  } else {
    new_types.push_back(out.getType());
  }
  auto ln_op =
      rewriter.create<tpu::LayerNormOp>(op.getLoc(), new_types, opds, attrs);
  Value output = ln_op.getOutput();
  std::vector<NamedAttribute> binary_attrs;
  auto name = module::getName(op.getOutput()).str();
  if (need_split_affine && have_weight) {
    auto mul_loc = NameLoc::get(rewriter.getStringAttr(name + "_affine_mul"));
    output = rewriter
                 .create<tpu::MulOp>(mul_loc, new_types[0],
                                     ValueRange{output, weight}, binary_attrs)
                 .getOutput();
  }
  if (need_split_affine && have_bias) {
    auto add_loc = NameLoc::get(rewriter.getStringAttr(name + "_affine_add"));
    output = rewriter
                 .create<tpu::AddOp>(add_loc, new_types[0],
                                     ValueRange{output, bias}, binary_attrs)
                 .getOutput();
  }
  op.replaceAllUsesWith(output);
  rewriter.eraseOp(op);
  return;
}

void LayerNormLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::LayerNormOp op) const {
  LoweringLayerNorm(rewriter, op, rewriter.getF32Type());
}

void LayerNormLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::LayerNormOp op,
                                     bool asymmetric) const {
  if (!module::isCV184X() && !module::isSGTPUV8()) {
    LoweringLayerNorm(rewriter, op, rewriter.getF16Type());
  } else {
    LoweringLayerNorm(rewriter, op, rewriter.getBF16Type());
  }
}

void LayerNormLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::LayerNormOp op,
                                     bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void LayerNormLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::LayerNormOp op) const {
  if (module::isBM1688() || module::isSG2380()) {
    LoweringLayerNorm(rewriter, op, rewriter.getF32Type());
  } else {
    LoweringLayerNorm(rewriter, op, rewriter.getBF16Type());
  }
}

void LayerNormLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::LayerNormOp op) const {
  LoweringLayerNorm(rewriter, op, rewriter.getF16Type());
}

void LayerNormLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::LayerNormOp op) const {
  LoweringLayerNorm(rewriter, op, rewriter.getF16Type());
}

void LayerNormLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::LayerNormOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
