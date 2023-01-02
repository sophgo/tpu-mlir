//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-upsample"
namespace tpu_mlir {
namespace cv18xx {

bool UpsampleToDeconv(PatternRewriter &rewriter, top::UpsampleOp &op) {
  int64_t scale_h = op.getScaleH();
  int64_t scale_w = op.getScaleW();

  if (scale_h >= 16 || scale_w >= 16) {
    return false;
  }
  auto input_shape = module::getShape(op.getInput());
  int64_t g = input_shape[1];
  int64_t oc = input_shape[1] / g;
  int64_t ic = input_shape[1] / g;
  int64_t h = scale_h;
  int64_t w = scale_w;

  int64_t count = g * oc * ic * h * w;
  std::vector<float> filter(count, 1);
  std::vector<int64_t> filter_shape;
  if (g != 1) {
    filter_shape.emplace_back(g);
  }
  filter_shape.emplace_back(oc);
  filter_shape.emplace_back(ic);
  filter_shape.emplace_back(h);
  filter_shape.emplace_back(w);

  std::string op_name = module::getName(op.getOutput()).str();
  auto filter_type = RankedTensorType::get(filter_shape, rewriter.getF32Type());
  auto filter_op =
      top::WeightOp::create(op, op_name + "filter", filter, filter_type);

  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(
      rewriter.getNamedAttr("kernel_shape", rewriter.getI64ArrayAttr({h, w})));
  attrs.emplace_back(rewriter.getNamedAttr(
      "strides", rewriter.getI64ArrayAttr({scale_h, scale_w})));
  attrs.emplace_back(
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
  attrs.emplace_back(
      rewriter.getNamedAttr("dilations", rewriter.getI64ArrayAttr({1, 1})));
  attrs.emplace_back(
      rewriter.getNamedAttr("inserts", rewriter.getI64ArrayAttr({0, 0})));
  attrs.emplace_back(
      rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(g)));

  std::vector<Value> operands;
  operands.emplace_back(op.getInput());
  operands.emplace_back(filter_op);
  operands.emplace_back(module::getNoneOp(op));
  rewriter.replaceOpWithNewOp<top::DeconvOp>(
      op, op.getOutput().getType().cast<RankedTensorType>(), operands, attrs);
  return true;
}

void UpsampleLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::UpsampleOp op, bool asymmetric) const {
  if (!UpsampleToDeconv(rewriter, op)) {
    lowering_common_int8<tpu::UpsampleOp>(rewriter, op, asymmetric);
  }
}

void UpsampleLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::UpsampleOp op) const {
  if (!UpsampleToDeconv(rewriter, op)) {
    lowering_common_bf16<tpu::UpsampleOp>(rewriter, op);
  }
}

} // namespace cv18xx
} // namespace tpu_mlir
