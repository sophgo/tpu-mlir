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

#define DEBUG_TYPE "lowering-conv"

namespace tpu_mlir {
namespace cv18xx {
// TODO use passes
static bool ConvertConv1d(PatternRewriter &rewriter, top::ConvOp op) {
  auto kernel = module::getI64Array(op.getKernelShape());
  if (kernel->size() != 1) {
    return false;
  }
  std::vector<int64_t> vfilterShape, vAttr;
  module::getShapeVec(op.getFilter(), vfilterShape);
  vfilterShape.push_back(1);
  auto new_type = RankedTensorType::get(vfilterShape, rewriter.getF32Type());
  op.getFilter().setType(new_type);

  // update kernel_shape
  auto kernel_shape = module::getI64Array(op.getKernelShape());
  vAttr = {kernel_shape->at(0), 1};
  op->setAttr("kernel_shape", rewriter.getI64ArrayAttr(vAttr));
  // update strides
  vAttr.clear();
  auto strides = module::getI64Array(op.getStrides());
  vAttr = {strides->at(0), 1};
  op->setAttr("strides", rewriter.getI64ArrayAttr(vAttr));
  // update pads
  vAttr.clear();
  auto pads_v = module::getI64Array(op.getPads());
  vAttr = {pads_v->at(0), 0, pads_v->at(1), 0};
  op->setAttr("pads", rewriter.getI64ArrayAttr(vAttr));
  // update dilations
  vAttr.clear();
  auto dilations = module::getI64Array(op.getDilations(), kernel_shape->size(), 1);
  vAttr = {dilations->at(0), 1};
  op->setAttr("dilations", rewriter.getI64ArrayAttr(vAttr));
  auto convOp = rewriter.create<top::ConvOp>(op->getLoc(), op->getResultTypes(),
                                             op->getOperands(), op->getAttrs());
  rewriter.replaceOp(op, {convOp.getOutput()});
  return true;
}

static bool ConvertDilation(PatternRewriter &rewriter, top::ConvOp op,
                            const conv_attr_t &attr) {
  const int DILATION_H_MAX = 15;
  const int DILATION_W_MAX = 15;
  if (attr.dh <= DILATION_H_MAX && attr.dw <= DILATION_W_MAX)
    return false;
  // filter
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  std::vector<int64_t> filterShape;
  module::getShapeVec(op.getFilter(), filterShape);
  int64_t oc = 0;
  int64_t ic = 0;
  int64_t kh = 0;
  int64_t kw = 0;
  if (filterShape.size() == 4) {
    oc = filterShape[0];
    ic = filterShape[1];
    kh = filterShape[2];
    kw = filterShape[3];
  } else if (filterShape.size() == 5) {
    // g, oc/g, ic/g, kh, kw
    oc = filterShape[0] * filterShape[1];
    ic = filterShape[2];
    kh = filterShape[3];
    kw = filterShape[4];
  } else {
    llvm_unreachable("Not support now.");
  }

  int insertNumH = 0;
  int insertNumW = 0;
  int newDilationH = attr.dh;
  int newDilationW = attr.dw;
  while (1) {
    insertNumH++;
    newDilationH = (attr.dh - 1 - insertNumH) / (insertNumH + 1) + 1;
    if (((attr.dh - 1 - insertNumH) % (insertNumH + 1) == 0) &&
        newDilationH < DILATION_H_MAX)
      break;
  }

  if (attr.dw > 1) {
    while (1) {
      insertNumW++;
      newDilationW = (attr.dw - 1 - insertNumW) / (insertNumW + 1) + 1;
      if (((attr.dw - 1 - insertNumW) % (insertNumW + 1) == 0) &&
          newDilationW < DILATION_W_MAX)
        break;
    }
  }

  int k_ext_h = (insertNumH + 1) * (kh - 1) + 1;
  int k_ext_w = (insertNumW + 1) * (kw - 1) + 1;
  filterShape[2] = k_ext_h;
  filterShape[3] = k_ext_w;
  auto filterSize = oc * ic * k_ext_h * k_ext_w;
  std::vector<float> newFilter(filterSize, 0);
  for (int i = 0; i < oc * ic; i++) {
    for (int j = 0; j < kh; j++) {
      for (int k = 0; k < kw; k++) {
        auto old_offset = i * kh * kw + j * kw + k;
        auto new_offset = i * k_ext_h * k_ext_w +
                          j * (insertNumH + 1) * k_ext_w + k * (insertNumW + 1);
        newFilter[new_offset] = filter_f32->data()[old_offset];
      }
    }
  }

  // update filter op
  auto new_type = RankedTensorType::get(filterShape, rewriter.getF32Type());
  auto new_filter_op =
      top::WeightOp::create(op, "dilation", newFilter, new_type);
  op->setOperand(1, new_filter_op);
  // update convOp attr
  std::vector<int64_t> new_kernel_shape, new_dilations;
  auto kernel_shape = module::getI64Array(op.getKernelShape());
  auto dilations = module::getI64Array(op.getDilations(), kernel_shape->size(), 1);
  new_kernel_shape.assign(kernel_shape->begin(), kernel_shape->end());
  new_dilations.assign(dilations->begin(), dilations->end());
  auto kernel_size = new_kernel_shape.size();
  new_kernel_shape[kernel_size - 2] = k_ext_h;
  new_kernel_shape[kernel_size - 1] = k_ext_w;

  new_dilations[kernel_size - 2] = newDilationH;
  new_dilations[kernel_size - 1] = newDilationW;

  op->setAttr("kernel_shape", rewriter.getI64ArrayAttr(new_kernel_shape));
  op->setAttr("dilations", rewriter.getI64ArrayAttr(new_dilations));
  auto convOp = rewriter.create<top::ConvOp>(op->getLoc(), op->getResultTypes(),
                                             op->getOperands(), op->getAttrs());
  rewriter.replaceOp(op, {convOp.getOutput()});
  return true;
}

static bool ConvertPading(PatternRewriter &rewriter, top::ConvOp op,
                          const conv_attr_t &attr) {
  // deal with pad > 16
  bool insert_pad = false;
  auto kernel_size = module::getI64Array(op.getKernelShape())->size();
  std::vector<int64_t> input_shape;
  module::getShapeVec(op.getInput(), input_shape);
  auto _pads = module::getI64Array(op.getPads());
  std::vector<int64_t> pad_v;
  std::vector<int64_t> new_pad_v(2 * input_shape.size(), 0);
  pad_v.assign(_pads->begin(), _pads->end());
  for (auto p : pad_v) {
    if (p > 15) {
      insert_pad = true;
      break;
    }
  }
  if (insert_pad && kernel_size > 3) {
    // todo pad 3d (padOp not support now)
    llvm_unreachable("Not supported now");
  }
  if (insert_pad) {
    if (kernel_size == 2) {
      if (attr.pht > 15) {
        assert(attr.pht == pad_v[0]);
        pad_v[0] = 0;
        new_pad_v[2] = attr.pht;
        input_shape[2] += attr.pht;
      }
      if (attr.pwl > 15) {
        assert(attr.pwl == pad_v[1]);
        pad_v[1] = 0;
        new_pad_v[3] = attr.pwl;
        input_shape[3] += attr.pwl;
      }
      if (attr.phb > 15) {
        assert(attr.phb == pad_v[2]);
        pad_v[2] = 0;
        new_pad_v[6] = attr.phb;
        input_shape[2] += attr.phb;
      }
      if (attr.pwr > 15) {
        assert(attr.pwr == pad_v[3]);
        pad_v[3] = 0;
        new_pad_v[7] = attr.pwr;
        input_shape[3] += attr.pwr;
      }
    }
    if (kernel_size == 1) {
      if (attr.pht > 15) {
        assert(attr.pht == pad_v[0]);
        pad_v[0] = 0;
        new_pad_v[2] = attr.pht;
        input_shape[2] += attr.pht;
      }
      if (attr.phb > 15) {
        assert(attr.phb == pad_v[1]);
        pad_v[1] = 0;
        new_pad_v[5] = attr.phb;
        input_shape[2] += attr.phb;
      }
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "paddings", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{new_pad_v})));
    auto op_name = module::getName(op.getOperation()).str();
    auto loc = NameLoc::get(rewriter.getStringAttr(op_name + "_pad"));
    auto type = op.getInput().getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(input_shape, type.getElementType());
    auto padOp = rewriter.create<top::PadOp>(loc, new_type,
                                             ValueRange{
                                                 op.getInput(),
                                             },
                                             attrs);
    op->setAttr("pads", rewriter.getI64ArrayAttr(pad_v));
    op->setOperand(0, padOp);
    auto convOp = rewriter.create<top::ConvOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands(), op->getAttrs());
    rewriter.replaceOp(op, {convOp.getOutput()});
  }
  return insert_pad;
}

void ConvLowering::LoweringINT8(PatternRewriter &rewriter, top::ConvOp op,
                                bool asymmetric) const {
  // for convert from hsigmoid/hswish
  if (!module::isCalibratedType(op.getOutput()) &&
      !module::isUniformQuantized(op.getOutput())) {
    LoweringBF16(rewriter, op);
    return;
  }
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  auto attr = op.parseParam();
  if (ConvertConv1d(rewriter, op)) {
    return;
  }
  if (ConvertPading(rewriter, op, attr)) {
    return;
  }
  if (ConvertDilation(rewriter, op, attr)) {
    return;
  }
  double in_thr, out_thr;
  in_thr = module::getThreshold(op.getInput());
  out_thr = module::getThreshold(op.getOutput());
  // filter
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();

  i32_array_t bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  auto filter_i8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  if (attr.has_bias) {
    auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  }
  std::vector<int64_t> rshift_v;
  std::vector<int64_t> multiplier_v;
  int64_t multiplier, rshift;
  int inner_dim = filter_f32->size() / attr.oc;
  // per-channel
  for (int c = 0; c < attr.oc; c++) { // per-channel quantize
    float *p_filter = filter_f32->data() + c * inner_dim;
    float w_max = findMaxabs(p_filter, inner_dim);
    double qscale = getQscaleForFilter(w_max, out_thr, in_thr);
    if (qscale >= 1) {
      // Now cv18xx not support lshift, if qscale > 1, rshift <= 0 not working
      // now we fix threshold_w to limit value qscale = (thr_w * thr_x) / (127.0
      // * thr_y) thr_w = qscale * 127.0 * thr_y / thr_x qscale = 0.99999999
      qscale = 0.999999;
      LLVM_DEBUG(llvm::errs()
                     << "WARNING: adjust threshold_w for qscale"
                     << ", qscale_filter = " << qscale << ", max_filter[" << c
                     << "] = " << qscale * 127 * out_thr / in_thr << "\n";);
    }
    if (attr.has_bias) {
      float b_max = fabs(bias_fp32->data()[c]);
      double qscale_bias = getQscaleForBias(b_max, out_thr);
      if (qscale_bias > qscale) {
        LLVM_DEBUG(llvm::errs() << "WARNING: adjust qscale for bias"
                                << ", qscale_filter = " << qscale
                                << ", qscale_bias = " << qscale_bias << "\n";);
        if (qscale_bias >= 1) {
          // prevent for auto tuning
          LLVM_DEBUG(llvm::errs()
                         << "WARNING:  qscale_bias are valid, keep org qscale"
                         << ", qscale_filter = " << qscale
                         << ", qscale_bias = " << qscale_bias << "\n";);
        } else {
          qscale = qscale_bias;
        }
      }
    }
    // decompose qscale into rshift and multiplier
    // QuantizeMultiplier(qscale, &multiplier, &rshift, false);
    getRShiftAndMultiplierFromQScale(qscale, &multiplier, &rshift, true);
    multiplier_v.push_back(multiplier);
    rshift_v.push_back(rshift);
    // quantize weight
    quantizeFilterRShiftAndMultiplier(
        p_filter, filter_i8->data() + c * inner_dim, inner_dim, out_thr, in_thr,
        rshift, multiplier, true);
    if (attr.has_bias) {
      quantizeBiasRShiftAndMultiplier(bias_fp32->data() + c,
                                      bias_int32->data() + c, 1, out_thr,
                                      rshift, multiplier, true);
    }
  }
  auto filter_type = op.getFilter().getType().cast<RankedTensorType>();
  auto new_type = RankedTensorType::get(filter_type.getShape(),
                                        rewriter.getIntegerType(8, true));
  auto new_filter =
      top::WeightOp::create(op, "filter_i8", *filter_i8, new_type);
  operands.push_back(new_filter);

  if (attr.has_bias) {
    auto new_type =
        RankedTensorType::get({1, attr.oc, 1, 1}, rewriter.getI32Type());
    auto new_bias =
        top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
    operands.push_back(new_bias);
  } else {
    operands.push_back(op.getBias()); // none
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto ctx = op->getContext();
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode", tpu::RequantModeAttr::get(ctx, tpu::RequantMode::Normal)));
  attrs.push_back(rewriter.getNamedAttr(
      "rshift", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v})));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{multiplier_v})));
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(attr.has_bias)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  auto newOp = rewriter.create<tpu::Conv2DOp>(op->getLoc(), newType,
                                              ArrayRef<Value>{operands},
                                              ArrayRef<NamedAttribute>{attrs});
  Value newValue = newOp.getOutput();
  rewriter.replaceOp(op, {newValue});
}

void ConvLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::ConvOp op) const {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  auto attr = op.parseParam();
  if (ConvertConv1d(rewriter, op)) {
    return;
  }
  if (ConvertPading(rewriter, op, attr)) {
    return;
  }
  if (ConvertDilation(rewriter, op, attr)) {
    return;
  }
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  operands.push_back(op.getInput());
  operands.push_back(filterOp.clone_bf16(op));
  operands.push_back(op.getBias());

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !op.getBias().getType().isa<mlir::NoneType>();
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto newType = getQuantBF16Type(op.getOutput());
  Value newValue;
  if (op.getKernelShape().size() == 1) {
    auto newOp =
        rewriter.create<tpu::Conv1DOp>(op->getLoc(), newType, operands, attrs);
    newValue = newOp.getOutput();
  } else if (op.getKernelShape().size() == 2) {
    auto newOp =
        rewriter.create<tpu::Conv2DOp>(op->getLoc(), newType, operands, attrs);
    newValue = newOp.getOutput();
  } else {
    auto newOp =
        rewriter.create<tpu::Conv3DOp>(op->getLoc(), newType, operands, attrs);
    newValue = newOp.getOutput();
  }
  rewriter.replaceOp(op, {newValue});
}

} // namespace cv18xx
} // namespace tpu_mlir
