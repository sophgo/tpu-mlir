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
  bool with_bias = !deconvOp.bias().getType().isa<mlir::NoneType>();
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));

  rewriter.replaceOpWithNewOp<tpu::DeconvOp>(op, deconvOp.output().getType(),
                                             operands, attrs);
}

void DeconvLowering::LoweringINT8(PatternRewriter &rewriter, top::DeconvOp op,
                                  bool asymmetric) const {
  if (asymmetric) {
    LoweringF32(rewriter, op);
    return;
  }
  deconv_attr_t param;
  op.parseParam(&param);
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.input());
  // in/out scale/zp
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(op.input(), in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(op.output(), out_scale, out_zp, asymmetric);
  // filter
  auto filterOp = cast<top::WeightOp>(op.filter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  float fmax, fmin;
  findMinMax(filter_f32->data(), filter_f32->size(), &fmin, &fmax);
  bool fsign = (fmin < 0 || param.with_bias == true);
  float fqmax = fsign ? 127 : 255;
  std::shared_ptr<std::vector<double>> weight_scale_v;
  if (filterOp.weight_scale().has_value() && weight_scale_v->size()) {
    weight_scale_v = Module::getF64Array(filterOp.weight_scale().value());
  }

  std::shared_ptr<std::vector<int32_t>> bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  auto filter_i8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  auto filter_u8 = std::make_shared<std::vector<uint8_t>>(filter_f32->size());
  if (param.with_bias) {
    auto biasOp = cast<top::WeightOp>(op.bias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  } else if (in_zp) {
    bias_int32 = std::make_shared<std::vector<int32_t>>(param.oc, 0);
  }

  double scale_w;
  int inner_dim = filter_f32->size() / param.oc;
  for (int c = 0; c < param.oc; c++) { // per-channel量化
    float *p_filter = filter_f32->data() + c * inner_dim;
    if (filterOp.weight_scale().has_value()) {
      scale_w = weight_scale_v->data()[c];
    } else {
      float w_max = findMaxabs(p_filter, inner_dim);
      scale_w = std::max(w_max / fqmax, 1e-5f);
    }
    if (fsign) {
      for (int t = 0; t < inner_dim; t++) {
        filter_i8->data()[c * inner_dim + t] =
            Quant::to_int8(p_filter[t] / scale_w);
      }
    } else {
      for (int t = 0; t < inner_dim; t++) {
        filter_u8->data()[c * inner_dim + t] =
            Quant::to_uint8(p_filter[t] / scale_w);
      }
    }

    double bias_w_xz = 0;
    if (in_zp) {
      for (int t = 0; t < inner_dim; t++) {
        bias_w_xz += filter_i8->data()[c * inner_dim + t] * in_zp;
      }
    }

    if (param.with_bias) {
      bias_int32->data()[c] =
          std::round(bias_fp32->data()[c] / (scale_w * in_scale) - bias_w_xz);
    } else if (in_zp) {
      bias_int32->data()[c] = std::round(-bias_w_xz);
    }
  }
  param.with_bias = (bias_int32 != nullptr);

  auto filter_type = op.filter().getType().cast<RankedTensorType>();
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
  if (param.with_bias) {
    auto new_type =
        RankedTensorType::get({1, param.oc, 1, 1}, rewriter.getI32Type());
    auto new_bias =
        top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
    operands.push_back(new_bias);
  } else {
    operands.push_back(op.bias()); // none
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  std::string deconv_name = Module::getName(op.getOperation()).str() + "_i32";
  auto name = rewriter.getStringAttr(deconv_name);
  attrs.push_back(rewriter.getNamedAttr("with_bias",
                                        rewriter.getBoolAttr(param.with_bias)));
  auto deconvType = RankedTensorType::get(
      {param.n, param.oc, param.oh, param.ow}, rewriter.getI32Type());
  auto deconvOp = rewriter.create<tpu::DeconvOp>(NameLoc::get(name), deconvType,
                                                 operands, attrs);

  auto rqType = getQuantInt8Type(op.output(), asymmetric);

  auto quant_int32 = std::make_shared<std::vector<int32_t>>(param.oc * 3, 0);
  int int32_multiplier, rshift;
  for (int c = 0; c < param.oc; c++) { // per-channel量化
    float *p_filter = filter_f32->data() + c * inner_dim;
    float w_max = findMaxabs(p_filter, inner_dim);
    double scale_w = std::max(w_max / fqmax, 1e-5f);
    double scale_f = scale_w * in_scale / out_scale;
    get_scale_and_shift(scale_f, int32_multiplier, rshift, 32);
    quant_int32->data()[3 * c] = int32_multiplier;
    quant_int32->data()[3 * c + 1] = -rshift;
    quant_int32->data()[3 * c + 2] = out_zp;
  }
  auto new_quant_type =
      RankedTensorType::get({1, param.oc, 1, 3}, rewriter.getI32Type());
  auto new_quant =
      top::WeightOp::create(op, "quant_int32", *quant_int32, new_quant_type);

  attrs.clear();
  auto ctx = op->getContext();
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode", tpu::RequantModeAttr::get(ctx, tpu::RequantMode::Normal)));
  operands.clear();
  operands.push_back(deconvOp.output());
  operands.push_back(new_quant);
  auto rqOp = rewriter.create<tpu::RequantIntAxisOp>(op->getLoc(), rqType,
                                                     operands, attrs);
  rewriter.replaceOp(op, {rqOp.output()});
}

void DeconvLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::DeconvOp op) const {
  std::vector<Value> operands;
  auto filterOp = cast<top::WeightOp>(op.filter().getDefiningOp());
  operands.push_back(op.input());
  operands.push_back(filterOp.clone_bf16(op));
  operands.push_back(op.bias());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !op.bias().getType().isa<mlir::NoneType>();
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto newType = getQuantBF16Type(op.output());
  rewriter.replaceOpWithNewOp<tpu::DeconvOp>(op, newType, operands, attrs);
}

void DeconvLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::DeconvOp op) const {
  std::vector<Value> operands;
  auto filterOp = cast<top::WeightOp>(op.filter().getDefiningOp());
  operands.push_back(op.input());
  operands.push_back(filterOp.clone_f16(op));
  operands.push_back(op.bias());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !op.bias().getType().isa<mlir::NoneType>();
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto newType = getQuantF16Type(op.output());
  rewriter.replaceOpWithNewOp<tpu::DeconvOp>(op, newType, operands, attrs);
}

void DeconvLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::DeconvOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
