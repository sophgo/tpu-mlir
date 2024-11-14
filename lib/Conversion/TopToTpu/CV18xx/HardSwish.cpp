//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-HardSwish"
namespace tpu_mlir {
namespace cv18xx {
static inline double hswish(double x) {
  return x * std::max(0.0, std::min(1.0, x / 6 + 0.5));
}
void HardSwishLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::HardSwishOp op,
                                     bool asymmetric) const {
  // LoweringBF16(rewriter, op);
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    [](double val) { return hswish(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void HardSwishLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::HardSwishOp op) const {

  double alpha = 1.0 / 6.0;
  double beta = 0.5;
  auto input_val = op.getInput();
  auto input_shape = module::getShape(input_val);
  int64_t c = input_shape[1];
  // input_shape is the same with output_shape
  auto newType = RankedTensorType::get(input_shape, rewriter.getF32Type());
  // auto newType = getQuantBF16Type(op.getOutput());
  std::string name = module::getName(op.getResult()).str();
  auto none = module::getNoneOp(op);

  // convert HardSwish to scale(depthwise conv) + clip + eltwise mul
  // 1.depthwise conv
  rewriter.setInsertionPointAfterValue(input_val);
  std::vector<Value> conv_operands;
  std::string conv_name = name + "_scale";
  auto conv_loc = NameLoc::get(rewriter.getStringAttr(conv_name));
  std::vector<float> conv_fileter(c, alpha);
  std::vector<float> conv_bias(c, beta);
  auto filter_type = RankedTensorType::get({c, 1, 1, 1}, rewriter.getF32Type());
  auto bias_type = RankedTensorType::get({c}, rewriter.getF32Type());
  conv_operands.emplace_back(input_val);
  conv_operands.emplace_back(none);
  conv_operands.emplace_back(none);
  std::vector<NamedAttribute> conv_attrs;
  conv_attrs.emplace_back(
      rewriter.getNamedAttr("kernel_shape", rewriter.getI64ArrayAttr({1, 1})));
  conv_attrs.emplace_back(
      rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));
  conv_attrs.emplace_back(
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
  conv_attrs.emplace_back(
      rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(c)));
  // conv_attrs.emplace_back(rewriter.getNamedAttr("with_bias",
  // rewriter.getBoolAttr(true))); auto conv_op =
  // rewriter.create<tpu::Conv2DOp>(conv_loc, newType, conv_operands,
  // conv_attrs);
  auto conv_op = rewriter.create<top::ConvOp>(conv_loc, newType, conv_operands,
                                              conv_attrs);
  auto fliter_weight = top::WeightOp::create(conv_op, conv_name + "_filter",
                                             conv_fileter, filter_type);
  auto bias_weight =
      top::WeightOp::create(conv_op, conv_name + "_bias", conv_bias, bias_type);
  conv_op.setOperand(1, fliter_weight);
  conv_op.setOperand(2, bias_weight);
  auto conv_out = conv_op.getOutput();

  // 2.clip op
  rewriter.setInsertionPointAfterValue(conv_out);
  std::vector<Value> clip_operands;
  std::string clip_name = name + "_clip";
  auto clip_loc = NameLoc::get(rewriter.getStringAttr(clip_name));
  clip_operands.emplace_back(conv_out);
  std::vector<NamedAttribute> clip_attrs;
  clip_attrs.emplace_back(
      rewriter.getNamedAttr("min", rewriter.getF64FloatAttr(0.0)));
  clip_attrs.emplace_back(
      rewriter.getNamedAttr("max", rewriter.getF64FloatAttr(1.0)));
  auto clip_op = rewriter.create<top::ClipOp>(clip_loc, newType, clip_operands,
                                              clip_attrs);
  auto clip_out = clip_op.getOutput();

  // 3.eltwise mul
  // rewriter.setInsertionPointAfter(clip_out);
  auto caliType = getQuantBF16Type(op.getOutput());
  std::vector<Value> mul_operands;
  mul_operands.emplace_back(input_val);
  mul_operands.emplace_back(clip_out);
  std::vector<NamedAttribute> mul_attrs;
  mul_attrs.emplace_back(
      rewriter.getNamedAttr("relu_limit", rewriter.getF64FloatAttr(-1.0)));
  rewriter.replaceOpWithNewOp<top::MulOp>(op.getOperation(), caliType,
                                          mul_operands, mul_attrs);
}
} // namespace cv18xx
} // namespace tpu_mlir
