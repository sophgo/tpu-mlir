//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct NormalizeConvert : public OpRewriterPatternEx<NormalizeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  NormalizeConvert(MLIRContext *context)
      : OpRewriterPatternEx<NormalizeOp>(context, "NormalizeConvert") {}
  LogicalResult matchAndRewriteImpl(NormalizeOp op,
                                    PatternRewriter &rewriter) const override {
    Value input_var = op->getOperand(0);
    std::string name = module::getName(op->getResult(0)).str();
    auto shape = module::getShape(op.getOperand(0));
    int64_t c = shape[1];
    auto weight_op = dyn_cast<top::WeightOp>(op.getScale().getDefiningOp());
    auto scale_data = weight_op.read_as_float();
    // input_shape is the same with output_shape
    auto result_type = RankedTensorType::get(shape, rewriter.getF32Type());
    auto none = module::getNoneOp(op);

    // separate Normalize op to below 6 ops.
    // Eltwise OP(power(x,2))-> Reduction(use conv now)-> Sqrt-> Div->Eltwise
    // OP(prod) ->Scale(by channel scale)

    // 1.Power OP
    rewriter.setInsertionPointAfterValue(input_var);
    std::string pow_name = name + "_square";
    auto loc = NameLoc::get(rewriter.getStringAttr(pow_name));
    std::vector<Value> operands;
    operands.push_back(input_var);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("exponent", rewriter.getF64FloatAttr(2.0)));
    auto pow_op = rewriter.create<PowOp>(loc, result_type, operands, attrs);
    auto pow_result_var = pow_op.getResult();

    // 2.Conv Op
    rewriter.setInsertionPointAfterValue(pow_result_var);
    std::string name_conv = name + "_conv";
    auto loc_conv = NameLoc::get(rewriter.getStringAttr(name_conv));
    std::vector<Value> operands_conv;
    std::vector<float> fliter_weight(c * c, 1.0);
    auto filter_type =
        RankedTensorType::get({c, c, 1, 1}, rewriter.getF32Type());
    operands_conv.push_back(pow_result_var);
    operands_conv.push_back(none);
    operands_conv.push_back(none);
    std::vector<NamedAttribute> attrs_conv;
    attrs_conv.push_back(rewriter.getNamedAttr(
        "kernel_shape", rewriter.getI64ArrayAttr({1, 1})));
    attrs_conv.push_back(
        rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));
    attrs_conv.push_back(
        rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
    auto conv_op = rewriter.create<ConvOp>(loc_conv, result_type, operands_conv,
                                           attrs_conv);
    auto conv_filter = WeightOp::create(conv_op, name_conv + "_filter",
                                        fliter_weight, filter_type);
    conv_op.setOperand(1, conv_filter);
    auto conv_result_var = conv_op.getResult();

    // 3.sqrt Op
    rewriter.setInsertionPointAfterValue(conv_result_var);
    std::string name_sqrt = name + "_sqrt";
    auto loc_sqrt = NameLoc::get(rewriter.getStringAttr(name_sqrt));
    std::vector<Value> operands_sqrt;
    operands_sqrt.push_back(conv_result_var);
    std::vector<NamedAttribute> attrs_sqrt;
    attrs_sqrt.push_back(
        rewriter.getNamedAttr("exponent", rewriter.getF64FloatAttr(0.5)));
    auto sqrt_op = rewriter.create<PowOp>(loc_sqrt, result_type, operands_sqrt,
                                          attrs_sqrt);
    auto sqrt_result_var = sqrt_op.getResult();

    // 4.Reciprocal Op
    rewriter.setInsertionPointAfterValue(sqrt_result_var);
    std::string name_recip = name + "_reciprocal";
    auto loc_recip = NameLoc::get(rewriter.getStringAttr(name_recip));
    std::vector<Value> operands_recip;
    operands_recip.push_back(sqrt_result_var);
    std::vector<NamedAttribute> attrs_recip;
    attrs_recip.push_back(
        rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(1.0)));
    auto recip_op = rewriter.create<ReciprocalOp>(loc_recip, result_type,
                                                  operands_recip, attrs_recip);
    auto recip_result_var = recip_op.getResult();

    // 5.Mul Op
    rewriter.setInsertionPointAfterValue(recip_result_var);
    std::string name_mul = name + "_mul";
    auto loc_mul = NameLoc::get(rewriter.getStringAttr(name_mul));
    std::vector<Value> operands_mul;
    operands_mul.push_back(input_var);
    operands_mul.push_back(recip_result_var);
    std::vector<NamedAttribute> attrs_mul;
    attrs_mul.push_back(
        rewriter.getNamedAttr("multiplier", rewriter.getSI32IntegerAttr(1)));
    attrs_mul.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(0)));
    auto mul_op =
        rewriter.create<MulOp>(loc_mul, result_type, operands_mul, attrs_mul);
    auto mul_result_var = mul_op.getResult();

    // 6.Scale Op (depthwise convolution)
    rewriter.setInsertionPointAfterValue(mul_result_var);
    std::string name_scale = name + "_scale";
    auto loc_scale = NameLoc::get(rewriter.getStringAttr(name_scale));
    std::vector<Value> operands_scale;
    std::vector<NamedAttribute> attrs_scale;
    std::vector<float> scale_weight(c);
    std::vector<float> scale_bias(c, 0.0);
    std::copy(scale_data->begin(), scale_data->end(), scale_weight.begin());
    auto scale_type =
        RankedTensorType::get({1, c, 1, 1}, rewriter.getF32Type());
    operands_scale.push_back(mul_result_var);
    operands_scale.push_back(none);
    operands_scale.push_back(none);
    auto scale_op = rewriter.create<ScaleOp>(loc_scale, result_type,
                                             operands_scale, attrs_scale);
    auto scale_weight_op = WeightOp::create(
        scale_op, name_scale + "_scaleWeight", scale_weight, scale_type);
    auto scale_bias_op = WeightOp::create(scale_op, name_scale + "_scaleBias",
                                          scale_bias, scale_type);
    scale_op.setOperand(1, scale_weight_op);
    scale_op.setOperand(2, scale_bias_op);
    // replace the origin NormalizeOp
    rewriter.replaceOp(op, scale_op);
    return success();
  }
};

void NormalizeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<NormalizeConvert>(context);
}
