//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Support/Float8.h"
namespace tpu_mlir {
namespace bm1684x {

void SubLowering::LoweringINT8(PatternRewriter &rewriter, top::SubOp op,
                               bool asymmetric) const {
  if (asymmetric) {
    lowering_common_f32<tpu::SubOp>(rewriter, op);
    return;
  }

  auto op_ = op.getOperation();
  std::vector<Value> operands;
  const int nInputs = op_->getNumOperands();
  std::vector<int64_t> rshift_v(nInputs);
  std::vector<int64_t> multiplier_v(nInputs, 1);
  int64_t o_zp;
  double o_scale;
  module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zp, asymmetric);
  auto coeff_v = module::getF64Array(op.getCoeff(), nInputs, 1.0);

  double scale;
  int64_t zeropoint;
  for (int i = 0; i < nInputs; i++) {
    auto input = op_->getOperand(i);
    int scalei = 1, shifti = 0;
    if (auto constOp = dyn_cast<top::WeightOp>(input.getDefiningOp())) {
      // constant tensor
      auto constF32 = constOp.read<float>();
      float fmax, fmin;
      findMinMax(constF32->data(), constF32->size(), &fmin, &fmax);
      fmax = std::max(fabs(fmax), fabs(fmin));
      bool cSign = (fmin < 0);
      auto filter_type = input.getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(filter_type.getShape(),
                                            rewriter.getIntegerType(8, cSign));
      // scale = fmax / fqmax;
      scale = o_scale; // Merge o_scale to Qconst, reducing multiple and shift
      if (cSign) {
        auto constI8 = std::make_shared<std::vector<int8_t>>(constF32->size());
        std::transform(constF32->begin(), constF32->end(), constI8->begin(),
                       [&](const float cf32) { return to_int8(cf32 / scale); });
        auto new_filter =
            top::WeightOp::create(constOp, "i8", *constI8, new_type);
        operands.push_back(new_filter);
      } else {
        auto constU8 = std::make_shared<std::vector<uint8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constU8->begin(),
            [&](const float cf32) { return to_uint8(cf32 / scale); });
        auto new_filter =
            top::WeightOp::create(constOp, "u8", *constU8, new_type);
        operands.push_back(new_filter);
      }
    } else {
      operands.push_back(input);
      module::getScaleAndZeroPoint(input, scale, zeropoint, asymmetric);
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
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multipliers", rewriter.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(rshift_v)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::SubOp>(op_, newType, operands, attrs);
}

void SubLowering::LoweringINT4(PatternRewriter &rewriter, top::SubOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SubLowering::LoweringF32(PatternRewriter &rewriter, top::SubOp op) const {
  lowering_common_f32<tpu::SubOp>(rewriter, op);
}

void SubLowering::LoweringBF16(PatternRewriter &rewriter, top::SubOp op) const {
  lowering_common_bf16<tpu::SubOp>(rewriter, op);
}

void SubLowering::LoweringF8(PatternRewriter &rewriter,
                             top::SubOp subOp) const {
  // UNREACHABLE_OP("Not Implemented", op);
  auto op = subOp.getOperation();
  const int numInputs = op->getNumOperands();
  if (module::getMode() == module::Mode::F8E5M2) {
    lowering_common_f8<tpu::SubOp>(rewriter, subOp, false, numInputs);
    return;
  }
  std::vector<Value> operands;
  std::vector<double> scale_v(numInputs);
  auto coeff_v = module::getF64Array(subOp.getCoeff(), numInputs, 1.0);
  auto qtype_out = module::getCalibratedType(subOp.getOutput());
  double out_scale = qtype_out.getMax();
  double cur_scale;
  Value cur_weight;
  for (int i = 0; i < numInputs; i++) {
    Value inValue = op->getOperand(i);
    if (auto weightOp = dyn_cast<top::WeightOp>(inValue.getDefiningOp())) {
      auto data = weightOp.read<float>();
      auto cnt = data->size();
#pragma omp parallel for schedule(static, omp_schedule(cnt))
      for (int j = 0; j < cnt; j++)
        data->at(j) = data->at(j) * get_f8e4m3_max() / out_scale;
      (void)weightOp.update(*data, cnt);
      cur_weight = weightOp.clone_f16(op);
      cur_scale = 1.;
      operands.push_back(cur_weight);
    } else {
      auto qtype_int = module::getCalibratedType(inValue);
      auto in_scale = qtype_int.getMax();
      cur_scale = coeff_v->at(i) * in_scale / out_scale;
      operands.push_back(inValue);
    }
    scale_v[i] = cur_scale;
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs())
    attrs.push_back(attr);
  attrs.push_back(
      rewriter.getNamedAttr("f8_scales", rewriter.getF64ArrayAttr(scale_v)));
  auto newType = getQuantF8E4M3Type(subOp.getOutput());
  rewriter.replaceOpWithNewOp<tpu::SubOp>(op, newType, operands, attrs);
}

void SubLowering::LoweringF16(PatternRewriter &rewriter, top::SubOp op) const {
  lowering_common_f16<tpu::SubOp>(rewriter, op);
}

//                / input0 -> dequant \
// quant sub ==> |                      sub -> requant
//                \ input1 -> dequant /
void SubLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::SubOp subOp) const {
  if (module::isUniformQuantized(subOp.getInputs()[0], subOp.getOutput()) ==
      false) {
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
  module::getScaleAndZeroPoint(subOp.getOutput(), o_scale, zeropoint, true);

  // generate quant param from given scale
  double scale, scale_max;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    module::getScaleAndZeroPoint(input, scale, zeropoint, true);
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
  bool is_const = false;
  int32_t const_val = 0;
  bool is_reverse = false;

  for (int i = 0; i < nInputs; ++i) {
    auto input = subOp->getOperand(i);
    if (module::isWeight(input)) {
      // do weight dequant in here
      int64_t num_elem = module::getNumElements(input);
      if (num_elem != 1) {
        auto new_input = do_weight_dequant(input, rewriter.getI32Type(),
                                           multiplier_v[i], shift_v[i], lshift);
        operands.push_back(new_input);
      } else {
        const_val =
            do_const_dequant(input, multiplier_v[i], shift_v[i], lshift);
        is_const = true;
        is_reverse = i == 0 ? true : false;
      }
    } else {
      // do dequant
      std::string name =
          module::getName(op).str() + "_dequant_" + std::to_string(i);
      auto name_loc = NameLoc::get(rewriter.getStringAttr(name));
      auto input_dequant =
          do_dequant(name_loc, input, rewriter.getI32Type(), multiplier_v[i],
                     shift_v[i], tpu::DequantMode::TFLite, lshift);
      operands.push_back(input_dequant);
    }
  }
  // // dequant left
  // std::string d0_name = module::getName(op).str() + "_dequant0";
  // auto name_loc_d0 = NameLoc::get(rewriter.getStringAttr(d0_name));
  // auto input0_dequant =
  //     do_dequant(name_loc_d0, subOp.getInputs()[0], rewriter.getI32Type(),
  //                multiplier_v[0], shift_v[0], tpu::DequantMode::TFLite,
  //                lshift);
  // // op->setOperand(0, input0_dequant);
  // operands.push_back(input0_dequant);
  // // dequant right
  // std::string d1_name = module::getName(op).str() + "_dequant1";
  // auto name_loc_d1 = NameLoc::get(rewriter.getStringAttr(d1_name));
  // auto input1_dequant =
  //     do_dequant(name_loc_d1, subOp.getInputs()[1], rewriter.getI32Type(),
  //                multiplier_v[1], shift_v[1], tpu::DequantMode::TFLite,
  //                lshift);
  // // op->setOperand(1, input1_dequant);
  // operands.push_back(input1_dequant);
  // sub
  std::string suffix = "_sub";
  std::string new_name = module::getName(op).str() + suffix;
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", subOp.getDoReluAttr()));
  auto newType = RankedTensorType::get(module::getShape(subOp.getOutput()),
                                       rewriter.getI32Type());
  // auto sub_quant = lowering_common<tpu::SubOp>(op, newType);
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  rewriter.setInsertionPointAfter(op);
  Value subout;
  if (is_const) {
    attrs.push_back(
        rewriter.getNamedAttr("is_reverse", rewriter.getBoolAttr(is_reverse)));
    attrs.push_back(rewriter.getNamedAttr("const_val",
                                          rewriter.getF64FloatAttr(const_val)));
    auto newOp =
        rewriter.create<tpu::SubConstOp>(name_loc, newType, operands, attrs);
    subout = newOp.getOutput();
  } else {
    auto newOp =
        rewriter.create<tpu::SubOp>(name_loc, newType, operands, attrs);
    subout = newOp.getOutput();
  }
  // requant to int8
  QuantizeMultiplier((scale_max * 2) / ((1 << lshift) * o_scale), &scalei,
                     &shifti);
  auto v = do_requant(op->getLoc(), subout, subOp.getOutput().getType(), true,
                      scalei, shifti, tpu::RequantMode::TFLite);
  rewriter.replaceOp(op, {v});
}

} // namespace bm1684x
} // namespace tpu_mlir
