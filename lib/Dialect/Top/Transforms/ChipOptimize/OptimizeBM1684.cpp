//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace tpu_mlir {
namespace bm1684 {
class ConvertUnsqueezeOp : public OpRewritePattern<top::UnsqueezeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::UnsqueezeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<top::ReshapeOp>(op, op.getOutput().getType(),
                                                op->getOperands(),
                                                std::vector<NamedAttribute>());
    return success();
  }
};

class ConvertSqueezeOp : public OpRewritePattern<top::SqueezeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::SqueezeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<top::ReshapeOp>(op, op.getOutput().getType(),
                                                op->getOperands(),
                                                std::vector<NamedAttribute>());
    return success();
  }
};

class ConvertScaleOp : public OpRewritePattern<top::ScaleOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto input_shape = module::getShape(op.getInput());
    if (module::isCalibratedType(op.getOutput().getType()) ||
        input_shape.size() > 4) {
      return failure();
    }
    auto cur_scale = dyn_cast<top::WeightOp>(op.getScale().getDefiningOp());
    auto cur_bias = dyn_cast<top::WeightOp>(op.getBias().getDefiningOp());
    if (!(cur_scale && cur_bias) || input_shape.size() < 3) {
      return failure();
    }
    int channel = cur_scale.getType().cast<RankedTensorType>().getNumElements();
    auto cur_scale_f32 = cur_scale.read<float>();
    auto cur_bias_f32 = cur_bias.read<float>();

    std::vector<float> new_scale_v(channel);
    std::vector<float> new_bias_v(channel);
    std::copy(cur_scale_f32->begin(), cur_scale_f32->end(),
              new_scale_v.begin());
    std::copy(cur_bias_f32->begin(), cur_bias_f32->end(), new_bias_v.begin());

    // scale to depthwise convolution
    NamedAttrList attrs;
    attrs.set("kernel_shape", rewriter.getI64ArrayAttr({1, 1}));
    attrs.set("strides", rewriter.getI64ArrayAttr({1, 1}));
    attrs.set("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0}));
    attrs.set("group", rewriter.getI64IntegerAttr(channel));
    attrs.set("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
    auto relu_limit = op.getReluLimit().convertToDouble();
    attrs.set("relu_limit", rewriter.getF64FloatAttr(relu_limit));

    auto filter_type =
        RankedTensorType::get({channel, 1, 1, 1}, rewriter.getF32Type());
    auto new_scale =
        top::WeightOp::create(op, "_to_weight", new_scale_v, filter_type);
    auto bias_type = RankedTensorType::get({channel}, rewriter.getF32Type());
    auto new_bias =
        top::WeightOp::create(op, "_to_bias", new_bias_v, bias_type);

    rewriter.replaceOpWithNewOp<top::ConvOp>(
        op, op.getResult().getType(),
        ValueRange{op.getInput(), new_scale, new_bias}, attrs);
    return success();
  }
};

class MergeScale2Conv : public OpRewritePattern<top::ScaleOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ScaleOp op,
                                PatternRewriter &rewriter) const override {
    if (module::isCalibratedType(op.getOutput().getType())) {
      return failure();
    }
    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->hasOneUse() || !isa<top::ConvOp>(formerOp)) {
      return failure();
    }
    auto conv_op = cast<top::ConvOp>(formerOp);
    if (conv_op.getDoRelu()) {
      return failure();
    }

    auto cur_scale_op = dyn_cast<top::WeightOp>(op.getScale().getDefiningOp());
    auto cur_bias_op = dyn_cast<top::WeightOp>(op.getBias().getDefiningOp());
    auto cur_scale_f32 = cur_scale_op.read<float>();
    auto cur_bias_f32 = cur_bias_op.read<float>();

    auto conv_weight_op =
        dyn_cast<top::WeightOp>(conv_op.getFilter().getDefiningOp());
    auto conv_bias_op =
        dyn_cast<top::WeightOp>(conv_op.getBias().getDefiningOp());

    int64_t oc, ic, kh, kw;
    module::getNCHW(conv_weight_op.getOutput(), oc, ic, kh, kw);

    // merge weight: weight = weight * cur_scale
    std::vector<float> conv_weight_v(oc * ic * kh * kw, 0);
    auto conv_weight_f32 = conv_weight_op.read<float>();
    for (int i = 0; i < oc; ++i) {
      for (int j = 0; j < kw * kh * ic; ++j) {
        conv_weight_v[i * ic * kh * kw + j] =
            conv_weight_f32->at(i * ic * kh * kw + j) * cur_scale_f32->at(i);
      }
    }
    // merge bias: bias = bias * cur_scale + cur_bias
    std::vector<float> conv_bias_v(oc, 0);
    if (conv_bias_op != nullptr) {
      auto conv_bias_f32 = conv_bias_op.read<float>();
      for (int i = 0; i < oc; ++i) {
        conv_bias_v[i] =
            conv_bias_f32->at(i) * cur_scale_f32->at(i) + cur_bias_f32->at(i);
      }
    } else {
      for (int i = 0; i < oc; ++i) {
        conv_bias_v[i] = cur_bias_f32->at(i);
      }
    }

    auto weight_type =
        RankedTensorType::get({oc, ic, kh, kw}, rewriter.getF32Type());
    auto conv_weight = top::WeightOp::create(conv_op, "merged_to_conv_weight",
                                             conv_weight_v, weight_type);
    auto bias_type = RankedTensorType::get({oc}, rewriter.getF32Type());
    auto conv_bias = top::WeightOp::create(conv_op, "merged_to_conv_bias",
                                           conv_bias_v, bias_type);
    conv_op->setOperand(1, conv_weight);
    conv_op->setOperand(2, conv_bias);
    conv_op.getOutput().setType(op.getOutput().getType());
    // update attrs
    double relu_limit = op.getReluLimit().convertToDouble();
    formerOp->setLoc(op.getLoc());
    formerOp->setAttr("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
    formerOp->setAttr("relu_limit", rewriter.getF64FloatAttr(relu_limit));

    // remove the scale Op
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

} // namespace bm1684

namespace top {
using namespace bm1684;
void populateOptimizeBM1684Patterns(RewritePatternSet *patterns) {
  // add bm1684 optimize here
  patterns->add<ConvertSqueezeOp, ConvertUnsqueezeOp, MergeScale2Conv,
                ConvertScaleOp>(patterns->getContext());
}

} // namespace top
} // namespace tpu_mlir
