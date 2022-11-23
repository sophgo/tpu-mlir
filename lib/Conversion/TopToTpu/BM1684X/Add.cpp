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

void AddLowering::LoweringINT8(PatternRewriter &rewriter, top::AddOp addOp,
                               bool asymmetric) const {
  if (asymmetric) {
    LoweringF32(rewriter, addOp);
    return;
  }
  auto op = addOp.getOperation();
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  std::vector<int64_t> rshift_v(nInputs);
  std::vector<int64_t> multiplier_v(nInputs, 1);
  int64_t o_zp;
  double o_scale;
  Quant::getScaleAndZeroPoint(addOp.output(), o_scale, o_zp, asymmetric);
  auto coeff_v = Module::getF64Array(addOp.coeff(), nInputs, 1.0);

  double scale;
  int64_t zeropoint;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    int scalei = 1, shifti = 0;
    if (auto constOp = dyn_cast<top::WeightOp>(input.getDefiningOp())) {
      // constant tensor
      auto constF32 = constOp.read<float>();
      float fmax, fmin;
      findMinMax(constF32->data(), constF32->size(), &fmin, &fmax);
      bool cSign = (fmin < 0);
      float fqmax = cSign ? 127 : 255;
      auto filter_type = input.getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(filter_type.getShape(),
                                            rewriter.getIntegerType(8, cSign));
      // scale = fmax / fqmax;
      scale = o_scale; // Merge o_scale to Qconst, reducing multiple and shift
      if (cSign) {
        auto constI8 = std::make_shared<std::vector<int8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constI8->begin(),
            [&](const float cf32) { return Quant::to_int8(cf32 / scale); });
        auto new_filter =
            top::WeightOp::create(constOp, "i8", *constI8, new_type);
        operands.push_back(new_filter);
      } else {
        auto constU8 = std::make_shared<std::vector<uint8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constU8->begin(),
            [&](const float cf32) { return Quant::to_uint8(cf32 / scale); });
        auto new_filter =
            top::WeightOp::create(constOp, "u8", *constU8, new_type);
        operands.push_back(new_filter);
      }
    } else {
      operands.push_back(input);
      Quant::getScaleAndZeroPoint(input, scale, zeropoint, asymmetric);
      auto scale_f = scale / o_scale;
      // get_scale_and_shift(coeff_v->at(i) * scale_f, scalei, shifti, 8);
      // "get_scale_and_shift_positive" use positive right_shift, left_shift
      // will be converted to the multiplier.
      get_scale_and_shift_positive(coeff_v->at(i) * scale_f, scalei, shifti, 8);
    }
    multiplier_v[i] = scalei;
    rshift_v[i] = shifti;
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", addOp.do_reluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multipliers", rewriter.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(rshift_v)));
  auto newType = Quant::getQuantInt8Type(addOp.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::AddOp>(op, newType, operands, attrs);
}

void AddLowering::LoweringF32(PatternRewriter &rewriter,
                              top::AddOp addOp) const {
  lowering_common_f32<tpu::AddOp>(rewriter, addOp.getOperation());
}

void AddLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::AddOp addOp) const {
  for (int i = 0, n = addOp.getNumOperands(); i < n; ++i) {
    if (auto constOp =
            dyn_cast<top::WeightOp>(addOp.getOperand(i).getDefiningOp())) {
      addOp.setOperand(i, constOp.clone_bf16(addOp));
    }
  }
  lowering_common_bf16<tpu::AddOp>(rewriter, addOp.getOperation());
}

void AddLowering::LoweringF16(PatternRewriter &rewriter,
                              top::AddOp addOp) const {
  for (int i = 0, n = addOp.getNumOperands(); i < n; ++i) {
    if (auto constOp =
            dyn_cast<top::WeightOp>(addOp.getOperand(i).getDefiningOp())) {
      addOp.setOperand(i, constOp.clone_f16(addOp));
    }
  }
  lowering_common_f16<tpu::AddOp>(rewriter, addOp.getOperation());
}

//                / input0 -> dequant \
// quant add ==> |                      add -> requant
//                \ input1 -> dequant /
void AddLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::AddOp addOp) const {
  if (Quant::isUniformQuantized(addOp.inputs()[0], addOp.output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  auto op = addOp.getOperation();
  const int nInputs = op->getNumOperands();
  assert(nInputs == 2); // TODO: nInput==1
  const int nTensors = nInputs + 1;
  const int lshift = 20; // TODO: lshift == 15 if input dtype is int16
  std::vector<int64_t> shift_v(nTensors);
  std::vector<int64_t> multiplier_v(nTensors, 1);
  std::vector<double> scale_v(nInputs);
  int64_t zeropoint;
  double o_scale;
  Quant::getScaleAndZeroPoint(addOp.output(), o_scale, zeropoint, true);

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
  bool is_const = false;
  int32_t const_val = 0;

  for (int i = 0; i < nInputs; ++i) {
    auto input = addOp->getOperand(i);
    if (isa<top::WeightOp>(input.getDefiningOp())) {
      // do dequant in here
      int64_t num_elem = Module::getNumElements(input);
      if (num_elem != 1) {
        auto new_input = do_weight_dequant(input, rewriter.getI32Type(),
                            multiplier_v[i], shift_v[i], lshift);
        operands.push_back(new_input);
      } else {
        const_val = do_const_dequant(input,  multiplier_v[i], shift_v[i], lshift);
        is_const = true;
      }
    } else {
      // do dequant
      std::string name = Module::getName(op).str() + "_dequant_" + std::to_string(i);
      auto name_loc = NameLoc::get(rewriter.getStringAttr(name));
      auto input_dequant =
          do_dequant(name_loc, input, rewriter.getI32Type(),
                    multiplier_v[i], shift_v[i], tpu::DequantMode::TFlite, lshift);
      operands.push_back(input_dequant);
    }
  }
  // // dequant left
  // std::string d0_name = Module::getName(op).str() + "_dequant0";
  // auto name_loc_d0 = NameLoc::get(rewriter.getStringAttr(d0_name));
  // auto input0_dequant =
  //     do_dequant(name_loc_d0, addOp.inputs()[0], rewriter.getI32Type(),
  //                multiplier_v[0], shift_v[0], tpu::DequantMode::TFlite, lshift);
  // // op->setOperand(0, input0_dequant);
  // operands.push_back(input0_dequant);
  // // dequant right
  // std::string d1_name = Module::getName(op).str() + "_dequant1";
  // auto name_loc_d1 = NameLoc::get(rewriter.getStringAttr(d1_name));
  // auto input1_dequant =
  //     do_dequant(name_loc_d1, addOp.inputs()[1], rewriter.getI32Type(),
  //                multiplier_v[1], shift_v[1], tpu::DequantMode::TFlite, lshift);
  // // op->setOperand(1, input1_dequant);
  // operands.push_back(input1_dequant);
  // add
  std::string suffix = "_add";
  std::string new_name = Module::getName(op).str() + suffix;
  std::vector<NamedAttribute> attrs;
  auto newType = RankedTensorType::get(Module::getShape(addOp.output()),
                                       rewriter.getI32Type());
  // auto add_quant = lowering_common<tpu::AddOp>(op, newType);
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  rewriter.setInsertionPointAfter(op);
  Value addout;
  if (is_const) {
    attrs.push_back(
        rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(const_val)));
    auto newOp = rewriter.create<tpu::AddConstOp>(name_loc, newType, operands, attrs);
    addout = newOp.output();
  } else {
    auto newOp = rewriter.create<tpu::AddOp>(name_loc, newType, operands, attrs);
    addout = newOp.output();
  }
  // requant to int8
  QuantizeMultiplier((scale_max * 2) / ((1 << lshift) * o_scale), &scalei,
                     &shifti);
  auto v = do_requant(op->getLoc(), addout, addOp.output().getType(),
                      true, scalei, shifti, tpu::RequantMode::TFlite);
  rewriter.replaceOp(op, {v});
}

} // namespace bm1684x
} // namespace tpu_mlir
