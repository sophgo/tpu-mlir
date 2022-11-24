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

void SubLowering::LoweringINT8(PatternRewriter &rewriter, top::SubOp subOp,
                               bool asymmetric) const {
  lowering_common_f32<tpu::SubOp>(rewriter, subOp.getOperation());
}

void SubLowering::LoweringF32(PatternRewriter &rewriter,
                              top::SubOp subOp) const {
  lowering_common_f32<tpu::SubOp>(rewriter, subOp.getOperation());
}

void SubLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::SubOp subOp) const {
  for (int i = 0, n = subOp.getNumOperands(); i < n; ++i) {
    if (auto constOp =
            dyn_cast<top::WeightOp>(subOp.getOperand(i).getDefiningOp())) {
      subOp.setOperand(i, constOp.clone_bf16(subOp));
    }
  }
  lowering_common_bf16<tpu::SubOp>(rewriter, subOp.getOperation());
}

void SubLowering::LoweringF16(PatternRewriter &rewriter,
                              top::SubOp subOp) const {
  for (int i = 0, n = subOp.getNumOperands(); i < n; ++i) {
    if (auto constOp =
            dyn_cast<top::WeightOp>(subOp.getOperand(i).getDefiningOp())) {
      subOp.setOperand(i, constOp.clone_f16(subOp));
    }
  }
  lowering_common_f16<tpu::SubOp>(rewriter, subOp.getOperation());
}

//                / input0 -> dequant \
// quant sub ==> |                      sub -> requant
//                \ input1 -> dequant /
void SubLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::SubOp subOp) const {
  if (Quant::isUniformQuantized(subOp.inputs()[0], subOp.output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  auto op = subOp.getOperation();
  const int nInputs = op->getNumOperands();
  assert(nInputs == 2); // TODO: nInput==1
  const int nTensors = nInputs + 1;
  const int lshift = 20; // TODO: lshift == 15 if input dtype is int16
  std::vector<int64_t> shift_v(nTensors);
  std::vector<int64_t> multiplier_v(nTensors, 1);
  std::vector<double> scale_v(nInputs);
  int64_t zeropoint;
  double o_scale;
  Quant::getScaleAndZeroPoint(subOp.output(), o_scale, zeropoint, true);

  // generate quant param from given scale
  double scale, scale_max;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    Quant::getScaleAndZeroPoint(input, scale, zeropoint, true);
    scale_v[i] = scale;
    if (i == 0) {
      scale_max = scale;
    } else {
      scale_max = scale > scale_max ? scale : scale_max;
    }
  }
  int64_t scalei, shifti;
  for (int i = 0; i < nInputs; i++) {
    auto scale_f = scale_v[i] / (scale_max * 2);
    QuantizeMultiplier(scale_f, &scalei, &shifti);
    multiplier_v[i] = scalei;
    shift_v[i] = shifti;
  }

  std::vector<Value> operands;
  auto ctx = op->getContext();

  // dequant left
  auto input0_dequant =
      do_dequant(subOp.inputs()[0], rewriter.getI32Type(), multiplier_v[0],
                 shift_v[0], tpu::DequantMode::TFlite, lshift);
  // op->setOperand(0, input0_dequant);
  operands.push_back(input0_dequant);
  // dequant right
  auto input1_dequant =
      do_dequant(subOp.inputs()[1], rewriter.getI32Type(), multiplier_v[1],
                 shift_v[1], tpu::DequantMode::TFlite, lshift);
  // op->setOperand(1, input1_dequant);
  operands.push_back(input1_dequant);
  // sub
  std::string suffix = "_sub";
  std::string new_name = Module::getName(op).str() + suffix;
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("multiplier", rewriter.getI64IntegerAttr(1)));
  attrs.push_back(
      rewriter.getNamedAttr("shift", rewriter.getI64IntegerAttr(0)));

  auto newType = RankedTensorType::get(Module::getShape(subOp.output()),
                                       rewriter.getI32Type());
  // auto sub_quant = lowering_common<tpu::SubOp>(op, newType);
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  rewriter.setInsertionPointAfter(op);
  auto newOp = rewriter.create<tpu::SubOp>(name_loc, newType, operands, attrs);
  // requant to int8
  QuantizeMultiplier((scale_max * 2) / ((1 << lshift) * o_scale), &scalei,
                     &shifti);
  auto v = do_requant(op->getLoc(), newOp.output(), subOp.output().getType(),
                      true, scalei, shifti, tpu::RequantMode::TFlite);
  rewriter.replaceOp(op, {v});
}

} // namespace bm1684x
} // namespace tpu_mlir
