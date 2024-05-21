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

static void Deconv_Lowering_F32(PatternRewriter &rewriter, top::DeconvOp op) {
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !module::isNone(op.getBias());
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  if (op.getKernelShape().size() == 2) {
    rewriter.replaceOpWithNewOp<tpu::DeconvOp>(op, op.getOutput().getType(),
                                               operands, attrs);
  } else if (op.getKernelShape().size() == 3) {
    rewriter.replaceOpWithNewOp<tpu::Deconv3DOp>(op, op.getOutput().getType(),
                                                 operands, attrs);
  } else {
    UNREACHABLE_OP("Not Implemented", op);
  }
}

void DeconvLowering::LoweringINT8(PatternRewriter &rewriter, top::DeconvOp op,
                                  bool asymmetric) const {
  if (op.getKernelShape().size() == 3) {
    Deconv_Lowering_F32(rewriter, op);
    return;
  }
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  std::vector<NamedAttribute> attrs;
  auto attr = op.parseParam();
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp, asymmetric);
  module::getScaleAndZeroPoint(op.getOutput(), out_scale, out_zp, asymmetric);
  auto filter_max = findMaxabs(filter_f32->data(), filter_f32->size());
  int rshift = calRightShiftNum(filter_max, in_scale, out_scale, BITS_INT8);
  if (rshift < 0) {
    // lowring as fp32
    LoweringF32(rewriter, op);
  } else {
    // lowring as quant
    std::shared_ptr<std::vector<int16_t>> bias_int16;
    if (attr.with_bias) {
      auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
      auto bias_fp32 = biasOp.read<float>();
      float bias_scale = 1.0 * (1 << rshift) / out_scale;
      int bias_len = bias_fp32->size();
      bias_int16 = std::make_shared<std::vector<int16_t>>(bias_len);
      float overflow_ratio = quantizeToInt16(
          bias_fp32->data(), bias_int16->data(), bias_len, bias_scale);

      while (overflow_ratio > 0.03 && rshift > 0) {
        rshift--;
        bias_scale = 1.0 * (1 << rshift) / out_scale;
        overflow_ratio = quantizeToInt16(bias_fp32->data(), bias_int16->data(),
                                         bias_len, bias_scale);
      }
    }
    std::vector<int64_t> rshift_v;
    std::vector<int64_t> multiplier_v;
    rshift_v.push_back(rshift);
    multiplier_v.push_back(1);

    float scale = 1.0 * (1 << rshift) * in_scale / out_scale;
    auto filter_int8 =
        std::make_shared<std::vector<int8_t>>(filter_f32->size());
    quantizeToInt8(filter_f32->data(), filter_int8->data(), filter_f32->size(),
                   scale);
    auto filter_type = op.getFilter().getType().cast<RankedTensorType>();
    auto new_type =
        RankedTensorType::get(filter_type.getShape(), rewriter.getI8Type());
    auto new_filter =
        top::WeightOp::create(op, "filter_int8", *filter_int8, new_type);
    operands.push_back(new_filter);
    // quant bias
    Value new_bias = op.getBias();
    if (attr.with_bias) {
      auto bias_type = op.getBias().getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(bias_type.getShape(),
                                            rewriter.getIntegerType(16));
      new_bias = top::WeightOp::create(op, "bias_int16", *bias_int16, new_type);
      operands.push_back(new_bias);
    } else {
      auto noneOp = module::getNoneOp(op);
      operands.push_back(noneOp);
    }

    for (auto &attr : op->getAttrs()) {
      attrs.push_back(attr);
    }
    attrs.push_back(rewriter.getNamedAttr(
        "rshift", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v})));
    attrs.push_back(rewriter.getNamedAttr(
        "multiplier",
        rewriter.getI64ArrayAttr(ArrayRef<int64_t>{multiplier_v})));
    attrs.push_back(rewriter.getNamedAttr(
        "quant_mode",
        tpu::RequantModeAttr::get(getContext(), tpu::RequantMode::OnlyShift)));
    attrs.push_back(rewriter.getNamedAttr(
        "with_bias", rewriter.getBoolAttr(attr.with_bias)));
    auto newType = getQuantInt8Type(op->getResult(0));
    if (op.getKernelShape().size() == 2) {
      rewriter.replaceOpWithNewOp<tpu::DeconvOp>(op, newType, operands, attrs);
    } else {
      UNREACHABLE_OP("Not Implemented", op);
    }
  }
}

void DeconvLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::DeconvOp op) const {
  Deconv_Lowering_F32(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
