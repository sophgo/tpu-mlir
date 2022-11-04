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
  Quant::getScaleAndZeroPoint(op.output(), scale_o, zp_o, asymmetric);

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
            [&](const float cf32) { return Quant::to_int8(cf32 / scale_i); });
        auto new_filter =
            top::WeightOp::create(constOp, "i8", *constI8, new_type);
        operands.push_back(new_filter);
      } else {
        auto constU8 = std::make_shared<std::vector<uint8_t>>(constF32->size());
        std::transform(
            constF32->begin(), constF32->end(), constU8->begin(),
            [&](const float cf32) { return Quant::to_uint8(cf32 / scale_i); });
        auto new_filter =
            top::WeightOp::create(constOp, "u8", *constU8, new_type);
        operands.push_back(new_filter);
      }
    } else {
      Quant::getScaleAndZeroPoint(input, scale_i, zp, asymmetric);
      operands.push_back(input);
    }
    scale *= scale_i;
  }

  scale /= scale_o;

  int multiplier;
  int rshift;
  get_scale_and_shift(scale, multiplier, rshift, 8);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.do_reluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));
  auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MulOp>(op, newType, operands, attrs);
}

void MulLowering::LoweringBF16(PatternRewriter &rewriter, top::MulOp op) const {
  for (int i = 0, n = op.getNumOperands(); i < n; ++i) {
    if (auto constOp =
            dyn_cast<top::WeightOp>(op.getOperand(i).getDefiningOp())) {
      op.setOperand(i, constOp.clone_bf16(op));
    }
  }
  lowering_common_bf16<tpu::MulOp>(rewriter, op);
}

void MulLowering::LoweringF16(PatternRewriter &rewriter, top::MulOp op) const {
  for (int i = 0, n = op.getNumOperands(); i < n; ++i) {
    if (auto constOp =
            dyn_cast<top::WeightOp>(op.getOperand(i).getDefiningOp())) {
      op.setOperand(i, constOp.clone_f16(op));
    }
  }
  lowering_common_f16<tpu::MulOp>(rewriter, op);
}

void MulLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::MulOp op) const {
  if (Quant::isUniformQuantized(op.inputs()[0], op.output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  assert(nInputs == 2);
  int64_t zeropoint;
  double scale, scale_mul;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    Quant::getScaleAndZeroPoint(input, scale, zeropoint, true);
    scale_mul *= scale;
  }
  Quant::getScaleAndZeroPoint(op.output(), scale, zeropoint, true);
  scale_mul = scale_mul / scale;

  int64_t multiplier;
  int64_t shift;
  QuantizeMultiplier(scale_mul, &multiplier, &shift);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.do_reluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(-shift)));
  auto newType = Quant::getQuantInt8Type(op.output(), true);
  rewriter.replaceOpWithNewOp<tpu::MulOp>(op, newType, operands, attrs);
}

} // namespace bm1684x
} // namespace tpu_mlir
