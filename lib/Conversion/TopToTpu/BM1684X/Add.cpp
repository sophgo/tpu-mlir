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

void AddTryLowering::Lowering(PatternRewriter &rewriter, top::AddOp op) const {
  auto opds = op->getOperands();
  auto all_shape = std::all_of(opds.begin(), opds.end(), [](Value opd) {
    return opd.getDefiningOp()->hasTrait<trait::ShapeProducer>();
  });
  if (all_shape) {
    std::vector<Value> operands;
    for (auto in : opds) {
      operands.push_back(in);
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("type", rewriter.getStringAttr("Add")));
    Type new_type = RankedTensorType::get(
        module::getShape(op.getOutput()),
        IntegerType::get(op.getOutput().getContext(), 32));
    rewriter.replaceOpWithNewOp<tpu::ShapeArithOp>(op, new_type, operands,
                                                   attrs);
  }
}

void AddLowering::LoweringINT8(PatternRewriter &rewriter, top::AddOp addOp,
                               bool asymmetric) const {
  auto op = addOp.getOperation();
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  std::vector<int64_t> rshift_v(nInputs);
  std::vector<int64_t> multiplier_v(nInputs, 1);
  int64_t o_zp;
  double o_scale;
  module::getScaleAndZeroPoint(addOp.getOutput(), o_scale, o_zp, asymmetric);
  auto coeff_v = module::getF64Array(addOp.getCoeff(), nInputs, 1.0);

  double scale;
  int64_t zeropoint;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    int scalei = 1, shifti = 0;

    if (!isa<BlockArgument>(input) &&
        isa<top::WeightOp>(input.getDefiningOp())) {
      // constant tensor
      auto constOp = dyn_cast<top::WeightOp>(input.getDefiningOp());
      auto constF32 = constOp.read<float>();
      float fmax, fmin;
      findMinMax(constF32->data(), constF32->size(), &fmin, &fmax);
      bool cSign = (fmin < 0);
      auto filter_type = input.getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(filter_type.getShape(),
                                            rewriter.getIntegerType(8, cSign));
      float absMax = findMaxabs(constF32->data(), constF32->size());
      if (cSign) {
        scale = (absMax / 127.0) / o_scale;
        get_scale_and_shift_positive(scale, scalei, shifti, 8);
        auto constI8 = std::make_shared<std::vector<int8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constI8->begin(),
            [&](const float cf32) { return to_int8(cf32 * 127.0 / absMax); });
        auto new_filter =
            top::WeightOp::create(constOp, "i8", *constI8, new_type);
        operands.push_back(new_filter);
      } else {
        scale = (absMax / 255.0) / o_scale;
        get_scale_and_shift_positive(scale, scalei, shifti, 8);
        auto constU8 = std::make_shared<std::vector<uint8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constU8->begin(),
            [&](const float cf32) { return to_uint8(cf32 * 255.0 / absMax); });
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
  attrs.push_back(rewriter.getNamedAttr("do_relu", addOp.getDoReluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multipliers", rewriter.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(rshift_v)));
  auto newType = getQuantInt8Type(addOp.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::AddOp>(op, newType, operands, attrs);
}

void AddLowering::LoweringINT4(PatternRewriter &rewriter, top::AddOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void AddLowering::LoweringF32(PatternRewriter &rewriter, top::AddOp op) const {
  for (uint32_t idx = 0; idx < op->getNumOperands(); idx++) {
    try_insert_host2device(op, idx);
  }
  lowering_common_f32<tpu::AddOp>(rewriter, op);
}

void AddLowering::LoweringBF16(PatternRewriter &rewriter, top::AddOp op) const {
  lowering_common_bf16<tpu::AddOp>(rewriter, op);
}

void AddLowering::LoweringF16(PatternRewriter &rewriter, top::AddOp op) const {
  lowering_common_f16<tpu::AddOp>(rewriter, op);
}

void AddLowering::LoweringF8(PatternRewriter &rewriter,
                             top::AddOp addOp) const {
  auto op = addOp.getOperation();
  const int numInputs = op->getNumOperands();
  if (module::getMode() == module::Mode::F8E5M2)
    lowering_common_f8<tpu::AddOp>(rewriter, addOp, false, numInputs);
  else if (module::getMode() == module::Mode::F8E4M3) {
    std::vector<Value> operands;
    std::vector<double> scale_v(numInputs);
    auto coeff_v = module::getF64Array(addOp.getCoeff(), numInputs, 1.0);
    auto qtype_out = module::getCalibratedType(addOp.getOutput());
    double out_scale = qtype_out.getMax();
    double cur_scale;
    Value cur_weight;
    for (int i = 0; i < numInputs; i++) {
      auto inValue = op->getOperand(i);
      if (!isa<BlockArgument>(inValue) &&
          isa<top::WeightOp>(inValue.getDefiningOp())) {
        auto weightOp = dyn_cast<top::WeightOp>(inValue.getDefiningOp());
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
    auto newType = getQuantF8E4M3Type(addOp.getOutput());
    rewriter.replaceOpWithNewOp<tpu::AddOp>(op, newType, operands, attrs);
  } else {
    llvm_unreachable("FIXME: not implement");
  }
}

//                / input0 -> dequant \
// quant add ==> |                      add -> requant
//                \ input1 -> dequant /
void AddLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::AddOp addOp) const {
  if (module::isUniformQuantized(addOp.getInputs()[0], addOp.getOutput()) ==
      false) {
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
  module::getScaleAndZeroPoint(addOp.getOutput(), o_scale, zeropoint, true);

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

  for (int i = 0; i < nInputs; ++i) {
    auto input = addOp->getOperand(i);
    if (module::isWeight(input)) {
      // do dequant in here
      int64_t num_elem = module::getNumElements(input);
      if (num_elem != 1) {
        auto new_input = do_weight_dequant(input, rewriter.getI32Type(),
                                           multiplier_v[i], shift_v[i], lshift);
        operands.push_back(new_input);
      } else {
        const_val =
            do_const_dequant(input, multiplier_v[i], shift_v[i], lshift);
        is_const = true;
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
  // add
  std::string suffix = "_add";
  std::string new_name = module::getName(op).str() + suffix;
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = RankedTensorType::get(module::getShape(addOp.getOutput()),
                                       rewriter.getI32Type());
  // auto add_quant = lowering_common<tpu::AddOp>(op, newType);
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  rewriter.setInsertionPointAfter(op);
  Value addout;
  if (is_const) {
    attrs.push_back(rewriter.getNamedAttr("const_val",
                                          rewriter.getF64FloatAttr(const_val)));
    auto newOp =
        rewriter.create<tpu::AddConstOp>(name_loc, newType, operands, attrs);
    addout = newOp.getOutput();
  } else {
    auto newOp =
        rewriter.create<tpu::AddOp>(name_loc, newType, operands, attrs);
    addout = newOp.getOutput();
  }
  // requant to int8
  QuantizeMultiplier((scale_max * 2) / ((1 << lshift) * o_scale), &scalei,
                     &shifti);
  auto v = do_requant(op->getLoc(), addout, addOp.getOutput().getType(), true,
                      scalei, shifti, tpu::RequantMode::TFLite);
  rewriter.replaceOp(op, {v});
}

} // namespace bm1684x
} // namespace tpu_mlir
