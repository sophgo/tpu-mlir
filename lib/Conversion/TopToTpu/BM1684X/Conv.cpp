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

static Value CreateConvOp(PatternRewriter &rewriter, int kernel_dims,
                          Location loc, Type type, std::vector<Value> &operands,
                          std::vector<NamedAttribute> &attrs) {
  switch (kernel_dims) {
  case 1: {
    auto newOp = rewriter.create<tpu::Conv1DOp>(loc, type, operands, attrs);
    return newOp.output();
  } break;
  case 2: {
    auto newOp = rewriter.create<tpu::Conv2DOp>(loc, type, operands, attrs);
    return newOp.output();
  } break;
  case 3: {
    auto newOp = rewriter.create<tpu::Conv3DOp>(loc, type, operands, attrs);
    return newOp.output();
  } break;
  default:
    llvm_unreachable("not support kernel dims");
  }
  return nullptr;
}

void ConvLowering::LoweringF32(PatternRewriter &rewriter,
                               top::ConvOp op) const {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !op.bias().getType().isa<mlir::NoneType>();
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto newValue = CreateConvOp(rewriter, op.kernel_shape().size(), op->getLoc(),
                               op.output().getType(), operands, attrs);
  rewriter.replaceOp(op, {newValue});
}

void ConvLowering::LoweringINT8(PatternRewriter &rewriter, top::ConvOp op,
                                bool asymmetric) const {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.input());
  conv_attr_t attr = {0};
  op.parseParam(&attr);
  // in/out scale/zp
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(op.input(), in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(op.output(), out_scale, out_zp, asymmetric);
  // filter
  auto filterOp = cast<top::WeightOp>(op.filter().getDefiningOp());
  bool is_quantized; // Make it uninitialized intentionally。
  if (filterOp.getResult()
          .getType()
          .dyn_cast<RankedTensorType>()
          .getElementType()
          .dyn_cast<IntegerType>()) {
    is_quantized = true;
  } else {
    is_quantized = false;
  }
  assert(!is_quantized ||
         (is_quantized && filterOp.weight_scale().has_value()) &&
             "Conv quant weight input must have scale attribute.");
  auto filter_f32 = is_quantized ? nullptr : filterOp.read<float>();
  auto filter_size = filterOp.getResult()
                         .getType()
                         .dyn_cast<RankedTensorType>()
                         .getNumElements();
  float fmax, fmin;
  if (filter_f32)
    findMinMax(filter_f32->data(), filter_size, &fmin, &fmax);
  bool fsign = (fmin < 0 || attr.has_bias == true ||
                is_quantized); // We only support signed qdq quant now.
  float fqmax = fsign ? 127 : 255;
  std::shared_ptr<std::vector<double>> weight_scale_v;
  if (filterOp.weight_scale().has_value()) {
    weight_scale_v = Module::getF64Array(filterOp.weight_scale().value());
  }

  std::shared_ptr<std::vector<int32_t>> bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  auto filter_i8 = std::make_shared<std::vector<int8_t>>(filter_size);
  auto filter_u8 = std::make_shared<std::vector<uint8_t>>(filter_size);
  if (attr.has_bias) {
    auto biasOp = cast<top::WeightOp>(op.bias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  } else if (in_zp) {
    bias_int32 = std::make_shared<std::vector<int32_t>>(attr.oc, 0);
  }

  std::vector<int64_t> rshift_v;
  std::vector<int64_t> multiplier_v;
  double scale_w;
  int int32_multiplier, shift;
  int inner_dim = filter_size / attr.oc;
  for (int c = 0; c < attr.oc; c++) { // per-channel量化
    float *p_filter =
        is_quantized ? nullptr : (filter_f32->data() + c * inner_dim);
    if (filterOp.weight_scale().has_value() && weight_scale_v->size()) {
      scale_w = weight_scale_v->data()[c];
    } else {
      float w_max = findMaxabs(p_filter, inner_dim);
      scale_w = std::max(w_max / fqmax, 1e-5f);
    }
    double scale_f = scale_w * in_scale / out_scale;
    get_scale_and_shift(scale_f, int32_multiplier, shift, 32);
    multiplier_v.push_back(int32_multiplier);
    rshift_v.push_back(shift);

    if (!is_quantized) {
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
    }

    double bias_w_xz = 0;
    if (in_zp) {
      for (int t = 0; t < inner_dim; t++) {
        bias_w_xz += filter_i8->data()[c * inner_dim + t] * in_zp;
      }
    }

    if (attr.has_bias) {
      bias_int32->data()[c] =
          std::round(bias_fp32->data()[c] / (scale_w * in_scale) - bias_w_xz);
    } else if (in_zp) {
      bias_int32->data()[c] = std::round(-bias_w_xz);
    }
  }
  attr.has_bias = (bias_int32 != nullptr);

  auto filter_type = op.filter().getType().cast<RankedTensorType>();
  auto new_type = RankedTensorType::get(filter_type.getShape(),
                                        rewriter.getIntegerType(8, fsign));
  if (is_quantized) {
    operands.push_back(op.filter());
  } else if (fsign) {
    auto new_filter =
        top::WeightOp::create(op, "filter_i8", *filter_i8, new_type);
    operands.push_back(new_filter);
  } else {
    auto new_filter =
        top::WeightOp::create(op, "filter_u8", *filter_u8, new_type);
    operands.push_back(new_filter);
  }
  if (attr.has_bias) {
    std::vector<int64_t> bias_shape(Module::getShape(op.output()).size(), 1);
    bias_shape[1] = attr.oc;
    auto new_type = RankedTensorType::get(bias_shape, rewriter.getI32Type());
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
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(attr.has_bias)));

  bool output_int32 = false;
  if (Module::isBM1686(LoweringConfig::chip) || op.kernel_shape().size() == 3) {
    output_int32 = true;
  }
  if (output_int32) {
    // to int32, and then requant to int8
    auto convType = RankedTensorType::get(Module::getShape(op.output()),
                                          rewriter.getI32Type());
    auto conv_name = Module::getName(op.getOperation()).str() + "_int32";
    auto name_loc = NameLoc::get(rewriter.getStringAttr(conv_name));
    auto conv_out = CreateConvOp(rewriter, op.kernel_shape().size(), name_loc,
                                 convType, operands, attrs);
    // requant
    auto output_type = getQuantInt8Type(op.output(), asymmetric);
    std::vector<int32_t> quant(attr.oc * 3, 0);
    for (size_t i = 0; i < attr.oc; ++i) {
      quant[i * 3] = multiplier_v[i];
      quant[i * 3 + 1] = -rshift_v[i];
      quant[i * 3 + 2] = out_zp;
    }
    auto quant_type =
        RankedTensorType::get({1, attr.oc, 1, 1, 3}, rewriter.getI32Type());
    auto quant_value = top::WeightOp::create(op, "quant", quant, quant_type);
    auto newValue = do_requant(op->getLoc(), conv_out, quant_value, output_type,
                               true, tpu::RequantMode::Normal);
    rewriter.replaceOp(op, {newValue});
    return;
  }

  auto ctx = op->getContext();
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode", tpu::RequantModeAttr::get(ctx, tpu::RequantMode::Normal)));
  attrs.push_back(rewriter.getNamedAttr(
      "rshift", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v})));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{multiplier_v})));
  auto newType = getQuantInt8Type(op.output(), asymmetric);

  auto newValue = CreateConvOp(rewriter, op.kernel_shape().size(), op->getLoc(),
                               newType, operands, attrs);
  rewriter.replaceOp(op, {newValue});
  return;
}

void ConvLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::ConvOp op) const {
  rewriter.setInsertionPointAfter(op);
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
  auto newValue = CreateConvOp(rewriter, op.kernel_shape().size(), op->getLoc(),
                               newType, operands, attrs);
  rewriter.replaceOp(op, {newValue});
}

void ConvLowering::LoweringF16(PatternRewriter &rewriter,
                               top::ConvOp op) const {
  rewriter.setInsertionPointAfter(op);
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
  auto newValue = CreateConvOp(rewriter, op.kernel_shape().size(), op->getLoc(),
                               newType, operands, attrs);
  rewriter.replaceOp(op, {newValue});
}

void ConvLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::ConvOp op) const {
  if (Quant::isUniformQuantized(op.input(), op.output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  conv_attr_t attr = {0};
  op.parseParam(&attr);
  auto input_qtype = Quant::getUniformQuantizedType(op.input());
  auto output_qtype = Quant::getUniformQuantizedType(op.output());
  auto filter_type = op.filter().getType().cast<RankedTensorType>();
  auto filter_qtype = filter_type.getElementType()
                          .dyn_cast<quant::UniformQuantizedPerAxisType>();
  int quant_size = 1;
  SmallVector<int64_t> shift(1);
  SmallVector<int64_t> multiplier(1);
  auto input_scale = input_qtype.getScale();
  auto output_scale = output_qtype.getScale();
  int32_t filter_zeroPoint;

  if (!filter_qtype) {
    auto filter_qtype = Quant::getUniformQuantizedType(op.filter());
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
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.input());
  auto filter_stype = Module::getStorageType(op.filter());
  auto filter_new_type =
      RankedTensorType::get(filter_type.getShape(), filter_stype);
  op.filter().setType(filter_new_type);
  operands.push_back(op.filter());

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  int32_t input_zeroPoint = input_qtype.getZeroPoint();
  bool with_bias = true;
  if (input_zeroPoint != 0) {
    // merge input_zeroPoint to bias
    auto filter_stype = Module::getStorageType(op.filter());
    std::shared_ptr<std::vector<int32_t>> bias_quant;
    std::shared_ptr<std::vector<int8_t>> filter_quant;
    filter_quant =
        cast<top::WeightOp>(op.filter().getDefiningOp()).read<int8_t>();
    if (attr.has_bias) {
      bias_quant =
          cast<top::WeightOp>(op.bias().getDefiningOp()).read<int32_t>();
    } else {
      // bias_quant->resize(attr.oc, 0);
      bias_quant = std::shared_ptr<std::vector<int32_t>>(
          new std::vector<int32_t>(attr.oc, 0));
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
  } else if (attr.has_bias) {
    auto bias_stype = Module::getStorageType(op.bias());
    auto bias_new_type =
        RankedTensorType::get(Module::getShape(op.bias()), bias_stype);
    op.bias().setType(bias_new_type);
    operands.push_back(op.bias());
  } else {
    with_bias = false;
    operands.push_back(op.bias());
  }
  if (filter_zeroPoint)
    attrs.push_back(rewriter.getNamedAttr(
        "kernel_zp", rewriter.getI64IntegerAttr(filter_zeroPoint)));
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto newType = RankedTensorType::get(Module::getShape(op.output()),
                                       rewriter.getI32Type());
  auto new_name = Module::getName(op.getOperation()).str() + "_int32";
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));

  auto newValue = CreateConvOp(rewriter, op.kernel_shape().size(), name_loc,
                               newType, operands, attrs);

  // do requant
  if (quant_size == 1) {
    newValue =
        do_requant(op->getLoc(), newValue, op.output().getType(), true,
                   multiplier[0], shift[0], tpu::RequantMode::TFlite_Lshift);
  } else {
    std::vector<int32_t> quant(quant_size * 3, 0);
    for (int i = 0; i < quant_size; ++i) {
      quant[i * 3] = multiplier[i];
      quant[i * 3 + 1] = shift[i];
      quant[i * 3 + 2] = output_qtype.getZeroPoint();
    }
    std::vector<int64_t> quant_shape(Module::getShape(op.input()).size(), 1l);
    quant_shape[1] = quant_size;
    quant_shape.back() = 3;
    auto quant_type = RankedTensorType::get(quant_shape, rewriter.getI32Type());
    auto quantValue = top::WeightOp::create(op, "quant", quant, quant_type);
    newValue =
        do_requant(op->getLoc(), newValue, quantValue, op.output().getType(),
                   true, tpu::RequantMode::TFlite_Lshift);
  }
  rewriter.replaceOp(op, {newValue});
}

} // namespace bm1684x
} // namespace tpu_mlir
