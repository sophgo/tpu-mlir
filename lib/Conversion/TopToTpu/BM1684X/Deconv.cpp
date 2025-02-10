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

static Value CreateDeconvOp(PatternRewriter &rewriter, int64_t dims,
                            Location loc, Type type,
                            std::vector<Value> &operands,
                            std::vector<NamedAttribute> &attrs) {
  switch (dims) {
  case 1:
  case 2: {
    auto newOp = rewriter.create<tpu::DeconvOp>(loc, type, operands, attrs);
    return newOp.getOutput();
  } break;
  case 3: {
    auto newOp = rewriter.create<tpu::Deconv3DOp>(loc, type, operands, attrs);
    return newOp.getOutput();
  } break;
  default:
    llvm_unreachable("not support kernel dims");
  }
  return nullptr;
}

void DeconvLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::DeconvOp deconvOp) const {
  rewriter.setInsertionPointAfter(deconvOp);
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
  auto newValue = CreateDeconvOp(
      rewriter, deconvOp.getKernelShape().size(), deconvOp->getLoc(),
      deconvOp.getOutput().getType(), operands, attrs);
  rewriter.replaceOp(op, {newValue});
}
void DeconvLowering::LoweringINT4(PatternRewriter &rewriter, top::DeconvOp op,
                                  bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void DeconvLowering::LoweringINT8(PatternRewriter &rewriter, top::DeconvOp op,
                                  bool asymmetric) const {
  if (module::isMARS3() || module::isSGTPUV8()) {
    LoweringBF16(rewriter, op);
    return;
  }
  if (module::isWeight(op.getFilter()) == false) {
    LoweringF32(rewriter, op);
    return;
  }
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
  bool with_bias = param.with_bias;
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
  auto deconvType = RankedTensorType::get(module::getShape(op.getOutput()),
                                          rewriter.getI32Type());
  auto newValue =
      CreateDeconvOp(rewriter, op.getKernelShape().size(), NameLoc::get(name),
                     deconvType, operands, attrs);

  auto rqType = getQuantInt8Type(op.getOutput(), asymmetric);

  bool output_int32 = false;
  if (op.getKernelShape().size() == 3) {
    output_int32 = true;
  }

  if (output_int32) {
    // to int32, and then requant to int8
    [[maybe_unused]] auto name_loc =
        NameLoc::get(rewriter.getStringAttr(deconv_name));
  }

  int q_size = module::isBM1684X() ? 3 : 2;
  auto quant_int32 =
      std::make_shared<std::vector<int32_t>>(param.oc * q_size, 0);
  int int32_multiplier, rshift;
  for (int c = 0; c < param.oc; c++) { // per-channel quant
    float *p_filter = filter_f32->data() + c * inner_dim;
    float w_max = findMaxabs(p_filter, inner_dim);
    double scale_w = std::max(w_max / fqmax, 1e-5f);
    double scale_f = scale_w * in_scale / out_scale;
    get_scale_and_shift(scale_f, int32_multiplier, rshift, 32);
    if (module::isBM1684X()) {
      quant_int32->data()[3 * c] = int32_multiplier;
      quant_int32->data()[3 * c + 1] = -rshift;
      quant_int32->data()[3 * c + 2] = out_zp;
    } else {
      quant_int32->data()[2 * c] = int32_multiplier;
      quant_int32->data()[2 * c + 1] =
          ((-(int32_t)rshift) & 0xffff) | (((int32_t)out_zp & 0xffff) << 16);
    }
  }
  auto new_quant_type1d =
      RankedTensorType::get({1, param.oc, 1, 1, q_size}, rewriter.getI32Type());
  auto new_quant_type2d =
      RankedTensorType::get({1, param.oc, 1, q_size}, rewriter.getI32Type());
  auto new_quant_type =
      op.getKernelShape().size() == 3 ? new_quant_type1d : new_quant_type2d;
  auto new_quant =
      top::WeightOp::create(op, "quant_int32", *quant_int32, new_quant_type);

  attrs.clear();
  auto ctx = op->getContext();
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode",
      tpu::RequantModeAttr::get(ctx, tpu::RequantMode::MultiplierShift)));
  operands.clear();
  operands.push_back(newValue);
  operands.push_back(new_quant);
  auto rqOp = rewriter.create<tpu::RequantIntAxisOp>(op->getLoc(), rqType,
                                                     operands, attrs);
  rewriter.replaceOp(op, {rqOp.getOutput()});
}

void DeconvLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::DeconvOp op) const {
  if (module::isWeight(op.getFilter()) == false) {
    LoweringF32(rewriter, op);
    return;
  }

  std::vector<Value> operands;
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  rewriter.setInsertionPointAfter(op);
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
  auto newValue = CreateDeconvOp(rewriter, op.getKernelShape().size(),
                                 op->getLoc(), newType, operands, attrs);
  rewriter.replaceOp(op, {newValue});
}

void DeconvLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::DeconvOp op) const {
  if (module::isWeight(op.getFilter()) == false) {
    LoweringF32(rewriter, op);
    return;
  }

  rewriter.setInsertionPointAfter(op);
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
  auto newValue = CreateDeconvOp(rewriter, op.getKernelShape().size(),
                                 op->getLoc(), newType, operands, attrs);
  rewriter.replaceOp(op, {newValue});
}

void DeconvLowering::LoweringF8(PatternRewriter &rewriter,
                                top::DeconvOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void DeconvLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::DeconvOp op) const {
  if (module::isUniformQuantized(op.getInput()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  bool out_i32 = module::isUniformQuantized(op.getOutput()) == false;
  auto p = op.parseParam();
  auto input_qtype = module::getUniformQuantizedType(op.getInput());
  auto filter_type = op.getFilter().getType().cast<RankedTensorType>();
  auto filter_qtype = filter_type.getElementType()
                          .dyn_cast<quant::UniformQuantizedPerAxisType>();
  int32_t filter_zeroPoint;

  if (!filter_qtype) {
    auto filter_qtype = module::getUniformQuantizedType(op.getFilter());
    filter_zeroPoint = filter_qtype.getZeroPoint();
  } else {
    filter_zeroPoint = filter_qtype.getZeroPoints()[0];
  }
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  operands.push_back(op.getFilter());

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  int32_t input_zeroPoint = input_qtype.getZeroPoint();
  bool with_bias = true;
  if (input_zeroPoint != 0) {
    // merge input_zeroPoint to bias
    auto filter_stype = module::getStorageType(op.getFilter());
    i32_array_t bias_quant;
    std::shared_ptr<std::vector<int8_t>> filter_quant;
    filter_quant =
        cast<top::WeightOp>(op.getFilter().getDefiningOp()).read<int8_t>();
    if (p.with_bias) {
      bias_quant =
          cast<top::WeightOp>(op.getBias().getDefiningOp()).read<int32_t>();
    } else {
      // bias_quant->resize(p.oc, 0);
      bias_quant = i32_array_t(new std::vector<int32_t>(p.oc, 0));
    }
    int64_t oc = filter_type.getShape()[0];
    int64_t kernel_size = filter_type.getNumElements() / oc;

    if (filter_stype.isUnsignedInteger(8)) {
      for (size_t oc_ind = 0; oc_ind < oc; ++oc_ind) {
        for (size_t kernel_ind = 0; kernel_ind < kernel_size; ++kernel_ind) {
          bias_quant->data()[oc_ind] -=
              input_zeroPoint *
              ((uint8_t)filter_quant->at(kernel_ind + oc_ind * kernel_size) -
               filter_zeroPoint);
        }
      }
    } else {
      for (size_t oc_ind = 0; oc_ind < oc; ++oc_ind) {
        for (size_t kernel_ind = 0; kernel_ind < kernel_size; ++kernel_ind) {
          bias_quant->data()[oc_ind] -=
              input_zeroPoint *
              (filter_quant->at(kernel_ind + oc_ind * kernel_size) -
               filter_zeroPoint);
        }
      }
    }
    auto bias_type = RankedTensorType::get({oc}, rewriter.getI32Type());
    auto new_bias =
        top::WeightOp::create(op, "_merge_bias", *bias_quant, bias_type);
    operands.push_back(new_bias);
  } else if (p.with_bias) {
    auto bias_stype = module::getStorageType(op.getBias());
    auto bias_new_type =
        RankedTensorType::get(module::getShape(op.getBias()), bias_stype);
    op.getBias().setType(bias_new_type);
    operands.push_back(op.getBias());
  } else {
    with_bias = false;
    operands.push_back(op.getBias());
  }
  if (filter_zeroPoint)
    attrs.push_back(rewriter.getNamedAttr(
        "kernel_zp", rewriter.getI64IntegerAttr(filter_zeroPoint)));
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto dims = module::getShape(op.getInput()).size() - 2;
  if (out_i32) {
    auto newValue = CreateDeconvOp(rewriter, dims, op->getLoc(),
                                   op.getOutput().getType(), operands, attrs);
    rewriter.replaceOp(op, {newValue});
    return;
  }

  auto newType = RankedTensorType::get(module::getShape(op.getOutput()),
                                       rewriter.getI32Type());
  auto new_name = module::getName(op.getOperation()).str() + "_int32";
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));

  auto newValue =
      CreateDeconvOp(rewriter, dims, name_loc, newType, operands, attrs);

  // generate requant param
  auto output_qtype = module::getUniformQuantizedType(op.getOutput());
  int quant_size = 1;
  SmallVector<int64_t> shift(1);
  SmallVector<int64_t> multiplier(1);
  auto input_scale = input_qtype.getScale();
  auto output_scale = output_qtype.getScale();

  if (!filter_qtype) {
    auto filter_qtype = module::getUniformQuantizedType(op.getFilter());
    filter_zeroPoint = filter_qtype.getZeroPoint();
    auto filter_scale = filter_qtype.getScale();
    const double effective_output_scale =
        input_scale * filter_scale / output_scale;
    QuantizeMultiplier(effective_output_scale, &multiplier[0], &shift[0]);
  } else {
    auto filter_scales = filter_qtype.getScales();
    filter_zeroPoint = filter_qtype.getZeroPoints()[0];
    quant_size = filter_scales.size();
    shift.resize(quant_size);
    multiplier.resize(quant_size);
    // tensorflow/lite/kernels/kernel_util.cc::PopulateConvolutionQuantizationParams
    // Populate multiplier and shift using affine quantization.
    for (auto filter : llvm::enumerate(filter_scales)) {
      const double effective_output_scale =
          input_scale * filter.value() / output_scale;
      QuantizeMultiplier(effective_output_scale, &multiplier[filter.index()],
                         &shift[filter.index()]);
    }
  }

  // do requant
  if (quant_size == 1) {
    newValue =
        do_requant(op->getLoc(), newValue, op.getOutput().getType(), true,
                   multiplier[0], shift[0], tpu::RequantMode::TFLite_LShift);
  } else {
    std::vector<int32_t> quant;
    std::vector<int64_t> quant_shape(module::getShape(op.getInput()).size(),
                                     1l);
    quant_shape[1] = quant_size;
    if (module::isBM1684X()) {
      quant.resize(quant_size * 3, 0);
      for (int i = 0; i < quant_size; ++i) {
        quant[i * 3] = multiplier[i];
        quant[i * 3 + 1] = shift[i];
        quant[i * 3 + 2] = output_qtype.getZeroPoint();
      }
      quant_shape.back() = 3;
    } else {
      quant.resize(quant_size * 2, 0);
      for (int i = 0; i < quant_size; ++i) {
        quant[i * 2] = multiplier[i];
        quant[i * 2 + 1] =
            (((int32_t)shift[i]) & 0xffff) |
            (((int32_t)output_qtype.getZeroPoint() & 0xffff) << 16);
      }
      quant_shape.back() = 2;
    }
    auto quant_type = RankedTensorType::get(quant_shape, rewriter.getI32Type());
    auto quantValue = top::WeightOp::create(op, "quant", quant, quant_type);
    newValue =
        do_requant(op->getLoc(), newValue, quantValue, op.getOutput().getType(),
                   true, tpu::RequantMode::TFLite_LShift);
  }
  rewriter.replaceOp(op, {newValue});
}

} // namespace bm1684x
} // namespace tpu_mlir
