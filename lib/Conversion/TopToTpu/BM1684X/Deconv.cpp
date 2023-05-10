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

void DeconvLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::DeconvOp deconvOp) const {
  auto op = deconvOp.getOperation();
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !module::isNone(deconvOp.getBias());
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));

  rewriter.replaceOpWithNewOp<tpu::DeconvOp>(op, deconvOp.getOutput().getType(),
                                             operands, attrs);
}
void DeconvLowering::LoweringINT4(PatternRewriter &rewriter, top::DeconvOp op,
                                  bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void DeconvLowering::LoweringINT8(PatternRewriter &rewriter, top::DeconvOp op,
                                  bool asymmetric) const {
  auto param = op.parseParam();
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  // in/out scale/zp
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp, asymmetric);
  module::getScaleAndZeroPoint(op.getOutput(), out_scale, out_zp, asymmetric);
  // filter
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  float fmax, fmin;
  findMinMax(filter_f32->data(), filter_f32->size(), &fmin, &fmax);
  bool with_bias = !module::isNone(op.getBias());
  bool fsign = (fmin < 0 || with_bias == true);
  float fqmax = fsign ? 127 : 255;
  f64_array_t weight_scale_v;
  if (filterOp.getScale().has_value() && weight_scale_v->size()) {
    weight_scale_v = module::getF64Array(filterOp.getScale().value());
  }

  i32_array_t bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  auto filter_i8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  auto filter_u8 = std::make_shared<std::vector<uint8_t>>(filter_f32->size());
  if (with_bias) {
    auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  } else if (in_zp) {
    bias_int32 = std::make_shared<std::vector<int32_t>>(param.oc, 0);
  }

  double scale_w;
  int inner_dim = filter_f32->size() / param.oc;
  for (int c = 0; c < param.oc; c++) { // per-channel量化
    float *p_filter = filter_f32->data() + c * inner_dim;
    if (filterOp.getScale().has_value()) {
      scale_w = weight_scale_v->data()[c];
    } else {
      float w_max = findMaxabs(p_filter, inner_dim);
      scale_w = std::max(w_max / fqmax, 1e-5f);
    }
    if (fsign) {
      for (int t = 0; t < inner_dim; t++) {
        filter_i8->data()[c * inner_dim + t] = to_int8(p_filter[t] / scale_w);
      }
    } else {
      for (int t = 0; t < inner_dim; t++) {
        filter_u8->data()[c * inner_dim + t] = to_uint8(p_filter[t] / scale_w);
      }
    }

    double bias_w_xz = 0;
    if (in_zp) {
      for (int t = 0; t < inner_dim; t++) {
        bias_w_xz += filter_i8->data()[c * inner_dim + t] * in_zp;
      }
    }

    if (with_bias) {
      bias_int32->data()[c] =
          std::round(bias_fp32->data()[c] / (scale_w * in_scale) - bias_w_xz);
    } else if (in_zp) {
      bias_int32->data()[c] = std::round(-bias_w_xz);
    }
  }
  with_bias = (bias_int32 != nullptr);

  auto filter_type = op.getFilter().getType().cast<RankedTensorType>();
  auto new_type = RankedTensorType::get(filter_type.getShape(),
                                        rewriter.getIntegerType(8, fsign));
  if (fsign) {
    auto new_filter =
        top::WeightOp::create(op, "filter_i8", *filter_i8, new_type);
    operands.push_back(new_filter);
  } else {
    auto new_filter =
        top::WeightOp::create(op, "filter_u8", *filter_u8, new_type);
    operands.push_back(new_filter);
  }
  if (with_bias) {
    auto new_type =
        RankedTensorType::get({1, param.oc, 1, 1}, rewriter.getI32Type());
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
  std::string deconv_name = module::getName(op.getOperation()).str() + "_i32";
  auto name = rewriter.getStringAttr(deconv_name);
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto deconvType = RankedTensorType::get(
      {param.n, param.oc, param.oh, param.ow}, rewriter.getI32Type());
  auto deconvOp = rewriter.create<tpu::DeconvOp>(NameLoc::get(name), deconvType,
                                                 operands, attrs);

  auto rqType = getQuantInt8Type(op.getOutput(), asymmetric);

  int q_size = module::isBM1686() ? 2 : 3;
  auto quant_int32 =
      std::make_shared<std::vector<int32_t>>(param.oc * q_size, 0);
  int int32_multiplier, rshift;
  for (int c = 0; c < param.oc; c++) { // per-channel quant
    float *p_filter = filter_f32->data() + c * inner_dim;
    float w_max = findMaxabs(p_filter, inner_dim);
    double scale_w = std::max(w_max / fqmax, 1e-5f);
    double scale_f = scale_w * in_scale / out_scale;
    get_scale_and_shift(scale_f, int32_multiplier, rshift, 32);
    if (module::isBM1686()) {
      quant_int32->data()[2 * c] = int32_multiplier;
      quant_int32->data()[2 * c + 1] =
          ((-(int32_t)rshift) & 0xffff) | (((int32_t)out_zp & 0xffff) << 16);

    } else {
      quant_int32->data()[3 * c] = int32_multiplier;
      quant_int32->data()[3 * c + 1] = -rshift;
      quant_int32->data()[3 * c + 2] = out_zp;
    }
  }
  auto new_quant_type =
      RankedTensorType::get({1, param.oc, 1, q_size}, rewriter.getI32Type());
  auto new_quant =
      top::WeightOp::create(op, "quant_int32", *quant_int32, new_quant_type);

  attrs.clear();
  auto ctx = op->getContext();
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode",
      tpu::RequantModeAttr::get(ctx, tpu::RequantMode::MultiplierShift)));
  operands.clear();
  operands.push_back(deconvOp.getOutput());
  operands.push_back(new_quant);
  auto rqOp = rewriter.create<tpu::RequantIntAxisOp>(op->getLoc(), rqType,
                                                     operands, attrs);
  rewriter.replaceOp(op, {rqOp.getOutput()});
}

void DeconvLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::DeconvOp op) const {
  std::vector<Value> operands;
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  operands.push_back(op.getInput());
  operands.push_back(filterOp.clone_bf16(op));
  operands.push_back(op.getBias());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !module::isNone(op.getBias());
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto newType = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::DeconvOp>(op, newType, operands, attrs);
}

void DeconvLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::DeconvOp op) const {
  std::vector<Value> operands;
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  operands.push_back(op.getInput());
  operands.push_back(filterOp.clone_f16(op));
  operands.push_back(op.getBias());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !module::isNone(op.getBias());
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto newType = getQuantF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::DeconvOp>(op, newType, operands, attrs);
}

void DeconvLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::DeconvOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
