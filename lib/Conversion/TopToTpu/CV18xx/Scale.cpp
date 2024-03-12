//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-relu"
namespace tpu_mlir {
namespace cv18xx {

static void ConvertToDw(PatternRewriter &rewriter, top::ScaleOp op) {

  auto cur_scale = dyn_cast<top::WeightOp>(op.getScale().getDefiningOp());
  auto cur_bias = dyn_cast<top::WeightOp>(op.getBias().getDefiningOp());
  if (!(cur_scale && cur_bias)) {
    llvm_unreachable("Not support now.");
  }
  int channel = cur_scale.getType().cast<RankedTensorType>().getNumElements();
  auto cur_scale_f32 = cur_scale.read<float>();
  auto cur_bias_f32 = cur_bias.read<float>();

  auto shape = module::getShape(op.getInput());
  assert(channel = shape[1]);
  int kernel_dim = shape.size() - 2;
  kernel_dim = std::max(kernel_dim, 2);

  std::vector<float> new_scale_v(channel);
  std::vector<float> new_bias_v(channel);
  std::copy(cur_scale_f32->begin(), cur_scale_f32->end(), new_scale_v.begin());
  std::copy(cur_bias_f32->begin(), cur_bias_f32->end(), new_bias_v.begin());

  // scale to depthwise convolution
  NamedAttrList attrs;
  auto kernel_shape = std::vector<int64_t>(kernel_dim, 1);
  auto strides = std::vector<int64_t>(kernel_dim, 1);
  auto pads = std::vector<int64_t>(kernel_dim * 2, 0);
  attrs.set("kernel_shape", rewriter.getI64ArrayAttr(kernel_shape));
  attrs.set("strides", rewriter.getI64ArrayAttr(strides));
  attrs.set("pads", rewriter.getI64ArrayAttr(pads));
  attrs.set("group", rewriter.getI64IntegerAttr(channel));
  attrs.set("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
  auto relu_limit = op.getReluLimit().convertToDouble();
  attrs.set("relu_limit", rewriter.getF64FloatAttr(relu_limit));

  std::vector<int64_t> filter_shape(kernel_dim + 2, 1);
  filter_shape[0] = channel;
  auto filter_type = RankedTensorType::get(filter_shape, rewriter.getF32Type());
  auto new_scale =
      top::WeightOp::create(op, "_to_weight", new_scale_v, filter_type);
  auto bias_type = RankedTensorType::get({channel}, rewriter.getF32Type());
  auto new_bias = top::WeightOp::create(op, "_to_bias", new_bias_v, bias_type);

  rewriter.replaceOpWithNewOp<top::ConvOp>(
      op, op.getResult().getType(),
      ValueRange{op.getInput(), new_scale, new_bias}, attrs);
}

void ScaleLowering::LoweringINT8(PatternRewriter &rewriter, top::ScaleOp op,
                                 bool asymmetric) const {
  assert(!asymmetric && "CV18xx not support asymmetric quantify");
  ConvertToDw(rewriter, op);
}

void ScaleLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::ScaleOp op) const {
  ConvertToDw(rewriter, op);
}
} // namespace cv18xx
} // namespace tpu_mlir
