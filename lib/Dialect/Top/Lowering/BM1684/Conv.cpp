//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

void top::ConvOp::lowering_int8_bm1684(PatternRewriter &rewriter) {
  auto op = getOperation();
  std::vector<Value> operands;
  operands.push_back(input());
  std::vector<NamedAttribute> attrs;
  conv_attr_t attr = {0};
  parseParam(&attr);
  auto filterOp = cast<top::WeightOp>(filter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  auto th_input = Quant::getThreshold(input());
  auto th_output = Quant::getThreshold(output());
  auto filter_max = findMaxabs(filter_f32->data(), filter_f32->size());
  int rshift =
      calRightShiftNum(filter_max, th_input, th_output, Quant::BITS_INT8);
  rshift = std::max(rshift, 0);
  std::shared_ptr<std::vector<int16_t>> bias_int16;
  if (attr.has_bias) {
    auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
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
  auto filter_type = filter().getType().cast<RankedTensorType>();
  auto new_type =
      RankedTensorType::get(filter_type.getShape(), rewriter.getI8Type());
  auto new_filter = WeightOp::create(op, "filter_int8", *filter_int8, new_type);
  operands.push_back(new_filter);
  Value new_bias = bias();
  if (attr.has_bias) {
    auto bias_type = bias().getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(bias_type.getShape(),
                                          rewriter.getIntegerType(16));
    new_bias = WeightOp::create(op, "bias_int16", *bias_int16, new_type);
  }
  operands.push_back(new_bias);
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr(
      "rshift", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v})));
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(attr.has_bias)));
  auto newType = Quant::getQuantInt8Type(output());
  rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(op, newType, operands, attrs);
}

void top::ConvOp::lowering_f32_bm1684(PatternRewriter &rewriter) {
  lowering_f32_bm1684x(rewriter);
}
