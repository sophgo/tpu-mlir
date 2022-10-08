//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void ConvLowering::LoweringF32(PatternRewriter &rewriter,
                               top::ConvOp op) const {
  auto ctx = getContext();
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !op.bias().getType().isa<mlir::NoneType>();
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));

  Value newValue;
  if (op.kernel_shape().size() == 1) {
    auto newOp = rewriter.create<tpu::Conv1DOp>(
        op->getLoc(), op.output().getType(), operands, attrs);
    newValue = newOp.output();
  } else if (op.kernel_shape().size() == 2) {
    auto newOp = rewriter.create<tpu::Conv2DOp>(
        op->getLoc(), op.output().getType(), operands, attrs);
    newValue = newOp.output();
  } else {
    auto newOp = rewriter.create<tpu::Conv3DOp>(
        op->getLoc(), op.output().getType(), operands, attrs);
    newValue = newOp.output();
  }
  rewriter.replaceOp(op, {newValue});
}

void ConvLowering::LoweringINT8(PatternRewriter &rewriter, top::ConvOp op,
                                bool asymmetric) const {
  std::vector<Value> operands;
  operands.push_back(op.input());
  std::vector<NamedAttribute> attrs;
  conv_attr_t attr = {0};
  op.parseParam(&attr);
  auto filterOp = cast<top::WeightOp>(op.filter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  auto th_input = Quant::getThreshold(op.input());
  auto th_output = Quant::getThreshold(op.output());
  auto filter_max = findMaxabs(filter_f32->data(), filter_f32->size());
  int rshift =
      calRightShiftNum(filter_max, th_input, th_output, Quant::BITS_INT8);
  rshift = std::max(rshift, 0);
  std::shared_ptr<std::vector<int16_t>> bias_int16;
  if (attr.has_bias) {
    auto biasOp = cast<top::WeightOp>(op.bias().getDefiningOp());
    auto bias_fp32 = biasOp.read<float>();
    float bias_scale = 1.0 * (1 << rshift) * Quant::QMAX_INT8 / th_output;
    int bias_len = bias_fp32->size();
    bias_int16 = std::make_shared<std::vector<int16_t>>(bias_len);
    float overflow_ratio = quantizeToInt16(
        bias_fp32->data(), bias_int16->data(), bias_len, bias_scale);

    int rightShiftDec = 2;
    while (overflow_ratio > 0.03 && rshift > 0) {
      rshift--;
      bias_scale = 1.0 * (1 << rshift) * Quant::QMAX_INT8 / th_output;
      overflow_ratio = quantizeToInt16(bias_fp32->data(), bias_int16->data(),
                                       bias_len, bias_scale);
      rightShiftDec--;
    }
  }
  std::vector<int64_t> rshift_v;
  rshift_v.push_back(rshift);
  std::vector<int64_t> multiplier_v;
  multiplier_v.push_back(1);
  float scale = 1.0 * (1 << rshift) * th_input / th_output;
  auto filter_int8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  quantizeToInt8(filter_f32->data(), filter_int8->data(), filter_f32->size(),
                 scale);
  auto filter_type = op.filter().getType().cast<RankedTensorType>();
  auto new_type =
      RankedTensorType::get(filter_type.getShape(), rewriter.getI8Type());
  auto new_filter =
      top::WeightOp::create(op, "filter_int8", *filter_int8, new_type);
  operands.push_back(new_filter);
  Value new_bias = op.bias();
  if (attr.has_bias) {
    auto bias_type = op.bias().getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(bias_type.getShape(),
                                          rewriter.getIntegerType(16));
    new_bias = top::WeightOp::create(op, "bias_int16", *bias_int16, new_type);
  }
  operands.push_back(new_bias);
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr(
      "rshift", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v})));
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(attr.has_bias)));
  auto newType = Quant::getQuantInt8Type(op.output());
  rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(op, newType, operands, attrs);
}

} // namespace bm1684
} // namespace tpu_mlir
