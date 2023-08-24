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

void DeformConv2DLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::DeformConv2DOp op) const {
  auto out_type = op.getOutput().getType();
  rewriter.setInsertionPointAfter(op);

  std::vector<Value> gather_operands, conv_operands;
  std::vector<NamedAttribute> cpu_params, gather_attrs, conv_attrs;

  cpu_params.push_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("deform_gather")));

  auto kernel = module::getI64Array(op.getKernelShape());
  gather_attrs.push_back(
      rewriter.getNamedAttr("kh", rewriter.getI64IntegerAttr(kernel->at(0))));
  gather_attrs.push_back(
      rewriter.getNamedAttr("kw", rewriter.getI64IntegerAttr(kernel->at(1))));
  conv_attrs.push_back(
      rewriter.getNamedAttr("kernel_shape", rewriter.getI64ArrayAttr({1, 1})));

  auto pads = module::getI64Array(op.getPads());
  gather_attrs.push_back(
      rewriter.getNamedAttr("pad_t", rewriter.getI64IntegerAttr(pads->at(0))));
  gather_attrs.push_back(
      rewriter.getNamedAttr("pad_l", rewriter.getI64IntegerAttr(pads->at(1))));
  gather_attrs.push_back(
      rewriter.getNamedAttr("pad_b", rewriter.getI64IntegerAttr(pads->at(2))));
  gather_attrs.push_back(
      rewriter.getNamedAttr("pad_r", rewriter.getI64IntegerAttr(pads->at(3))));
  conv_attrs.push_back(
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));

  auto strides = module::getI64Array(op.getStrides());
  gather_attrs.push_back(rewriter.getNamedAttr(
      "stride_h", rewriter.getI64IntegerAttr(strides->at(0))));
  gather_attrs.push_back(rewriter.getNamedAttr(
      "stride_w", rewriter.getI64IntegerAttr(strides->at(1))));
  conv_attrs.push_back(
      rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));

  auto dilations = module::getI64Array(op.getDilations(), kernel->size(), 1);
  gather_attrs.push_back(rewriter.getNamedAttr(
      "dilation_h", rewriter.getI64IntegerAttr(dilations->at(0))));
  gather_attrs.push_back(rewriter.getNamedAttr(
      "dilation_w", rewriter.getI64IntegerAttr(dilations->at(1))));
  conv_attrs.push_back(
      rewriter.getNamedAttr("dilations", rewriter.getI64ArrayAttr({1, 1})));

  int deform_groups = op.getDeformGroup();
  gather_attrs.push_back(rewriter.getNamedAttr(
      "deform_group", rewriter.getI64IntegerAttr(deform_groups)));

  bool use_mask = op.getUseMask();
  gather_attrs.push_back(
      rewriter.getNamedAttr("use_mask", rewriter.getBoolAttr(use_mask)));

  int groups = op.getGroup();
  conv_attrs.push_back(
      rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(groups)));

  bool with_bias = !module::isNone(op.getBias());
  conv_attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));

  gather_operands.push_back(op->getOperand(0)); // input
  gather_operands.push_back(op->getOperand(2)); // offset
  if (use_mask)
    gather_operands.push_back(op->getOperand(3)); // mask

  auto i_s = op.getInput().getType().cast<RankedTensorType>().getShape();
  const int conved_H = ((i_s[2] - (dilations->at(0) * (kernel->at(0) - 1) + 1) +
                         pads->at(0) + pads->at(2)) /
                            strides->at(0) +
                        1);
  const int conved_W = ((i_s[3] - (dilations->at(1) * (kernel->at(1) - 1) + 1) +
                         pads->at(1) + pads->at(3)) /
                            strides->at(1) +
                        1);

  auto shape_on = i_s[0];
  auto shape_oc = kernel->at(0) * kernel->at(1) * i_s[1];
  auto shape_oh = conved_H;
  auto shape_ow = conved_W;

  auto gather_type = module::getTypeLike(
      op.getOutput(), {shape_on, shape_oc, shape_oh, shape_ow});

  auto deform_gather_loc = module::getLocLike(op.getOutput(), "deform_gather");
  cpu_params.push_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(gather_attrs)));
  auto deform_gather_op = rewriter.create<tpu::GenericCpuOp>(
      deform_gather_loc, gather_type, gather_operands, cpu_params);

  conv_operands.push_back(deform_gather_op.getOutputs()[0]); // gather output
  conv_operands.push_back(op->getOperand(1));                // weight
  conv_operands.push_back(op->getOperand(4));                // bias

  auto conv_loc = op.getLoc();
  auto conv_op = rewriter.create<tpu::Conv2DOp>(conv_loc, out_type,
                                                conv_operands, conv_attrs);
  op.replaceAllUsesWith(conv_op.getOperation());
  rewriter.eraseOp(op);
}

void DeformConv2DLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::DeformConv2DOp op,
                                        bool asymmetric) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
