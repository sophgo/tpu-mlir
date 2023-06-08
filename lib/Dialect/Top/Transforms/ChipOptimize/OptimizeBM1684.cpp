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
  if (module::isCalibratedType(op.getOutput().getType()) || input_shape.size() > 4) {
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

} // namespace bm1684

namespace top {
using namespace bm1684;
void populateOptimizeBM1684Patterns(RewritePatternSet *patterns) {
  // add bm1684 optimize here
  patterns->add<
        ConvertSqueezeOp,
        ConvertUnsqueezeOp,
        ConvertScaleOp
        >(patterns->getContext());
}

} // namespace top
} // namespace tpu_mlir
