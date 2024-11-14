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

void DeformConv2DLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::DeformConv2DOp op) const {
  auto name = module::getName(op.getOutput());
  auto out_type = op.getOutput().getType();
  auto ele_type = out_type.cast<RankedTensorType>().getElementType();
  rewriter.setInsertionPointAfter(op);

  std::vector<Value> gather_operands, conv_operands;
  std::vector<NamedAttribute> gather_attrs, conv_attrs;

  auto kernel = module::getI64Array(op.getKernelShape());
  std::vector<int64_t> kernel_v = {kernel->at(0), kernel->at(1)};
  gather_attrs.push_back(rewriter.getNamedAttr(
      "kernel_shape", rewriter.getI64ArrayAttr(kernel_v)));
  conv_attrs.push_back(
      rewriter.getNamedAttr("kernel_shape", rewriter.getI64ArrayAttr({1, 1})));

  auto pads = module::getI64Array(op.getPads());
  std::vector<int64_t> pads_v = {pads->at(0), pads->at(1), pads->at(2),
                                 pads->at(3)};
  gather_attrs.push_back(
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(pads_v)));
  conv_attrs.push_back(
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));

  auto strides = module::getI64Array(op.getStrides());
  std::vector<int64_t> strides_v = {strides->at(0), strides->at(1)};
  gather_attrs.push_back(
      rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr(strides_v)));
  conv_attrs.push_back(
      rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));

  auto dilations = module::getI64Array(op.getDilations(), kernel->size(), 1);
  std::vector<int64_t> dilations_v = {dilations->at(0), dilations->at(1)};
  gather_attrs.push_back(rewriter.getNamedAttr(
      "dilations", rewriter.getI64ArrayAttr(dilations_v)));
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

  gather_operands.push_back(op->getOperand(0));                    // input
  gather_operands.push_back(op->getOperand(2));                    // offset
  gather_operands.push_back(op->getOperand(3));                    // mask
  gather_operands.push_back(module::getNoneOp(op.getOperation())); // buffer

  auto i_s = op.getInput().getType().cast<RankedTensorType>().getShape();
  const int conved_H = ((i_s[2] - (dilations_v[0] * (kernel_v[0] - 1) + 1) +
                         pads_v[0] + pads_v[2]) /
                            strides_v[0] +
                        1);
  const int conved_W = ((i_s[3] - (dilations_v[1] * (kernel_v[1] - 1) + 1) +
                         pads_v[1] + pads_v[3]) /
                            strides_v[1] +
                        1);

  auto shape_on = i_s[0];
  auto shape_oc = kernel_v[0] * kernel_v[1] * i_s[1];
  auto shape_oh = conved_H;
  auto shape_ow = conved_W;

  auto gather_type =
      RankedTensorType::get({shape_on, shape_oc, shape_oh, shape_ow}, ele_type);

  auto deform_gather_loc =
      NameLoc::get(rewriter.getStringAttr(name.str() + "_deform_gather"));
  auto deform_gather_op = rewriter.create<tpu::DeformGatherOp>(
      deform_gather_loc, gather_type, gather_operands, gather_attrs);

  conv_operands.push_back(deform_gather_op.getOutput()); // gather output
  conv_operands.push_back(op->getOperand(1));            // weight
  conv_operands.push_back(op->getOperand(4));            // bias

  auto conv_loc = op.getLoc();
  auto conv_op = rewriter.create<tpu::Conv2DOp>(conv_loc, out_type,
                                                conv_operands, conv_attrs);
  op.replaceAllUsesWith(conv_op.getOperation());
  rewriter.eraseOp(op);
}

void DeformConv2DLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::DeformConv2DOp op) const {
  LoweringF32(rewriter, op);
}

void DeformConv2DLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::DeformConv2DOp op) const {
  LoweringF32(rewriter, op);
}

void DeformConv2DLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::DeformConv2DOp op,
                                        bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void DeformConv2DLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::DeformConv2DOp op,
                                        bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void DeformConv2DLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::DeformConv2DOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void DeformConv2DLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::DeformConv2DOp op) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir
