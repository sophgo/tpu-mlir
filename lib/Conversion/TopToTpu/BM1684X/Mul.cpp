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

void MulLowering::LoweringF32(PatternRewriter &rewriter, top::MulOp op) const {
  lowering_common_f32<tpu::MulOp>(rewriter, op);
}
void MulLowering::LoweringINT4(PatternRewriter &rewriter, top::MulOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void MulLowering::LoweringINT8(PatternRewriter &rewriter, top::MulOp op,
                               bool asymmetric) const {
  if (asymmetric) {
    LoweringF32(rewriter, op);
    return;
  }
  const int nInputs = op->getNumOperands();
  std::vector<Value> operands;
  double scale = 1;
  int64_t zp_o = 0;
  double scale_o = 1;
  module::getScaleAndZeroPoint(op.getOutput(), scale_o, zp_o, asymmetric);

  double scale_i;
  int64_t zp;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    if (auto constOp = dyn_cast<top::WeightOp>(input.getDefiningOp())) {
      auto constF32 = constOp.read<float>();
      float fmax, fmin;
      findMinMax(constF32->data(), constF32->size(), &fmin, &fmax);
      bool cSign = (fmin < 0);
      float fqmax = cSign ? 127 : 255;
      auto filter_type = input.getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(filter_type.getShape(),
                                            rewriter.getIntegerType(8, cSign));
      scale_i = fmax / fqmax;
      if (cSign) {
        auto constI8 = std::make_shared<std::vector<int8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constI8->begin(),
            [&](const float cf32) { return to_int8(cf32 / scale_i); });
        auto new_filter =
            top::WeightOp::create(constOp, "i8", *constI8, new_type);
        operands.push_back(new_filter);
      } else {
        auto constU8 = std::make_shared<std::vector<uint8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constU8->begin(),
            [&](const float cf32) { return to_uint8(cf32 / scale_i); });
        auto new_filter =
            top::WeightOp::create(constOp, "u8", *constU8, new_type);
        operands.push_back(new_filter);
      }
    } else {
      module::getScaleAndZeroPoint(input, scale_i, zp, asymmetric);
      operands.push_back(input);
    }
    scale *= scale_i;
  }

  scale /= scale_o;

  int multiplier;
  int rshift;
  get_scale_and_shift(scale, multiplier, rshift, 8);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MulOp>(op, newType, operands, attrs);
}

void MulLowering::LoweringBF16(PatternRewriter &rewriter, top::MulOp op) const {
  lowering_common_bf16<tpu::MulOp>(rewriter, op);
}

void MulLowering::LoweringF16(PatternRewriter &rewriter, top::MulOp op) const {
  lowering_common_f16<tpu::MulOp>(rewriter, op);
}

void MulLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::MulOp op) const {
  if (module::isUniformQuantized(op.getInputs()[0], op.getOutput()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  assert(nInputs == 2);
  int64_t zeropoint;
  double scale, scale_mul = 1.f;
  bool is_const = false;
  float const_val = 0.f;
  std::string name = module::getName(op.getOutput()).str();
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    bool same = false;
    for (int j = 0; j < i; j++) {
      auto old = op->getOperand(j);
      if (old == input) {
        // same input
        same = true;
        operands.push_back(operands[j]);
        continue;
      }
    }
    if (same) {
      continue;
    }
    module::getScaleAndZeroPoint(input, scale, zeropoint, true);
    scale_mul *= scale;
    if (auto constOp = dyn_cast<top::WeightOp>(input.getDefiningOp())) {
      // do sub zp in here
      auto num_element = module::getNumElements(input);
      if (num_element == 1) {
        is_const = true;
        auto constF32 = constOp.read_as_float();
        const_val = constF32->data()[0] - zeropoint;
      } else {
        auto input_stype = module::getStorageType(input);
        auto input_sub_zp = std::make_shared<std::vector<int16_t>>(num_element);
        auto input_quant = constOp.read<int8_t>();
        if (input_stype.isUnsignedInteger(8)) {
          for (int64_t i = 0; i < num_element; ++i) {
            input_sub_zp->at(i) = (uint8_t)(input_quant->at(i)) - zeropoint;
          }
        } else {
          for (int64_t i = 0; i < num_element; ++i) {
            input_sub_zp->at(i) = input_quant->at(i) - zeropoint;
          }
        }
        auto new_type = RankedTensorType::get(module::getShape(input), rewriter.getI16Type());
        auto new_input = top::WeightOp::create(op, "_int16", *input_sub_zp, new_type);
        operands.push_back(new_input);
      }
    } else {
      auto input_sub_zp = do_binary_saclar<tpu::AddConstOp>(
          input, rewriter.getI16Type(), -zeropoint);
      operands.push_back(input_sub_zp);
    }
  }
  module::getScaleAndZeroPoint(op.getOutput(), scale, zeropoint, true);
  scale_mul = scale_mul / scale;

  int64_t multiplier;
  int64_t shift;
  QuantizeMultiplier(scale_mul, &multiplier, &shift);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  std::string suffix = "_mul";
  std::string new_name = name + suffix;
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  rewriter.setInsertionPointAfter(op);
  auto newType = RankedTensorType::get(module::getShape(op.getOutput()),
                                       rewriter.getI32Type());
  if (is_const == false) {
    auto newOp =
        rewriter.create<tpu::MulOp>(name_loc, newType, operands, attrs);
    // requant to int8
    auto v =
        do_requant(op->getLoc(), newOp.getOutput(), op.getOutput().getType(),
                   true, multiplier, shift, tpu::RequantMode::TFLite);
    rewriter.replaceOp(op, {v});
  } else {
    attrs.push_back(rewriter.getNamedAttr("const_val",
                                          rewriter.getF64FloatAttr(const_val)));
    auto newOp =
        rewriter.create<tpu::MulConstOp>(name_loc, newType, operands, attrs);
    // requant to int8
    auto v =
        do_requant(op->getLoc(), newOp.getOutput(), op.getOutput().getType(),
                   true, multiplier, shift, tpu::RequantMode::TFLite);
    rewriter.replaceOp(op, {v});
  }
}

} // namespace bm1684x
} // namespace tpu_mlir
