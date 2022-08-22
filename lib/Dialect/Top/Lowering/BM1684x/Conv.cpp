//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "mlir/IR/Location.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "../Lowering.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

Value top::ConvOp::lowering_int8_bm1684x(bool asymmetric) {
  auto op = getOperation();
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(input());
  conv_attr_t attr = {0};
  parseParam(&attr);
  // in/out scale/zp
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(input(), in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(output(), out_scale, out_zp, asymmetric);
  // filter
  auto filterOp = cast<top::WeightOp>(filter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  float fmax, fmin;
  findMinMax(filter_f32->data(), filter_f32->size(), &fmin, &fmax);
  bool fsign = (fmin < 0 || attr.has_bias == true);
  float fqmax = fsign ? 127 : 255;

  std::shared_ptr<std::vector<int32_t>> bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  auto filter_i8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  auto filter_u8 = std::make_shared<std::vector<uint8_t>>(filter_f32->size());
  if (attr.has_bias) {
    auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  } else if (in_zp) {
    bias_int32 = std::make_shared<std::vector<int32_t>>(attr.oc, 0);
  }

  std::vector<int64_t> rshift_v;
  std::vector<int64_t> multiplier_v;
  int int32_multiplier, shift;
  int inner_dim = filter_f32->size() / attr.oc;
  for (int c = 0; c < attr.oc; c++) { // per-channel量化
    float *p_filter = filter_f32->data() + c * inner_dim;
    float w_max = findMaxabs(p_filter, inner_dim);
    double scale_w = std::max(w_max / fqmax, 1e-5f);
    double scale_f = scale_w * in_scale / out_scale;
    get_scale_and_shift(scale_f, int32_multiplier, shift, 32);
    multiplier_v.push_back(int32_multiplier);
    rshift_v.push_back(shift);
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

    if (attr.has_bias) {
      bias_int32->data()[c] =
          std::round(bias_fp32->data()[c] / (scale_w * in_scale) - bias_w_xz);
    } else if (in_zp) {
      bias_int32->data()[c] = std::round(-bias_w_xz);
    }
  }
  attr.has_bias = (bias_int32 != nullptr);

  auto filter_type = filter().getType().cast<RankedTensorType>();
  auto new_type = RankedTensorType::get(filter_type.getShape(),
                                        builder.getIntegerType(8, fsign));
  if (fsign) {
    auto new_filter = WeightOp::create(op, "filter_i8", *filter_i8, new_type);
    operands.push_back(new_filter);
  } else {
    auto new_filter = WeightOp::create(op, "filter_u8", *filter_u8, new_type);
    operands.push_back(new_filter);
  }
  if (attr.has_bias) {
    auto new_type =
        RankedTensorType::get({1, attr.oc, 1, 1}, builder.getI32Type());
    auto new_bias = WeightOp::create(op, "bias_int32", *bias_int32, new_type);
    operands.push_back(new_bias);
  } else {
    operands.push_back(bias()); // none
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(
      builder.getNamedAttr("quant_mode", builder.getI64IntegerAttr(2)));
  attrs.push_back(builder.getNamedAttr(
      "rshift", builder.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v})));
  attrs.push_back(builder.getNamedAttr(
      "multiplier", builder.getI64ArrayAttr(ArrayRef<int64_t>{multiplier_v})));
  attrs.push_back(
      builder.getNamedAttr("with_bias", builder.getBoolAttr(attr.has_bias)));
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  auto newOp = builder.create<tpu::Conv2DOp>(op->getLoc(), newType,
                                             ArrayRef<Value>{operands},
                                             ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}

Value top::ConvOp::lowering_f32_bm1684x() {
  auto ctx = getContext();
  OpBuilder builder(ctx);
  auto op = getOperation();
  builder.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !bias().getType().isa<mlir::NoneType>();
  attrs.push_back(
      builder.getNamedAttr("with_bias", builder.getBoolAttr(with_bias)));

  auto newOp = builder.create<tpu::Conv2DOp>(op->getLoc(), output().getType(),
                                             ArrayRef<Value>{operands},
                                             ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}

Value top::ConvOp::lowering_f16_bm1684x() {
  auto ctx = getContext();
  OpBuilder builder(ctx);
  auto op = getOperation();
  builder.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  auto filterOp = cast<top::WeightOp>(filter().getDefiningOp());
  operands.push_back(input());
  operands.push_back(filterOp.clone_f16(op));
  operands.push_back(bias());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !bias().getType().isa<mlir::NoneType>();
  attrs.push_back(
      builder.getNamedAttr("with_bias", builder.getBoolAttr(with_bias)));
  auto tensor_type = output().getType().cast<RankedTensorType>();
  auto newType =
      RankedTensorType::get(tensor_type.getShape(), builder.getF16Type());
  auto newOp = builder.create<tpu::Conv2DOp>(op->getLoc(), newType,
                                             ArrayRef<Value>{operands},
                                             ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}

Value top::ConvOp::lowering_bf16_bm1684x() {
  auto ctx = getContext();
  OpBuilder builder(ctx);
  auto op = getOperation();
  builder.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  auto filterOp = cast<top::WeightOp>(filter().getDefiningOp());
  operands.push_back(input());
  operands.push_back(filterOp.clone_bf16(op));
  operands.push_back(bias());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !bias().getType().isa<mlir::NoneType>();
  attrs.push_back(
      builder.getNamedAttr("with_bias", builder.getBoolAttr(with_bias)));
  auto tensor_type = output().getType().cast<RankedTensorType>();
  auto newType =
      RankedTensorType::get(tensor_type.getShape(), builder.getBF16Type());
  auto newOp = builder.create<tpu::Conv2DOp>(op->getLoc(), newType,
                                             ArrayRef<Value>{operands},
                                             ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}

Value top::ConvOp::lowering_quant_bm1684x() {
  if (Quant::isUniformQuantized(input(), output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  conv_attr_t attr = {0};
  parseParam(&attr);
  auto input_qtype = Quant::getUniformQuantizedType(input());
  auto output_qtype = Quant::getUniformQuantizedType(output());
  auto filter_type = filter().getType().cast<RankedTensorType>();
  auto filter_qtype = filter_type.getElementType()
                          .dyn_cast<quant::UniformQuantizedPerAxisType>();
  if (!filter_qtype) {
    llvm_unreachable("TODO: per layer to support");
  }
  auto filter_scales = filter_qtype.getScales();

  SmallVector<int64_t> shift(filter_scales.size());
  SmallVector<int64_t> multiplier(filter_scales.size());

  // tensorflow/lite/kernels/kernel_util.cc::PopulateConvolutionQuantizationParams
  // Populate multiplier and shift using affine quantization.
  auto input_scale = input_qtype.getScale();
  auto output_scale = output_qtype.getScale();
  for (auto filter : llvm::enumerate(filter_scales)) {
    const double effective_output_scale =
        input_scale * filter.value() / output_scale;
    // int scale,shift;
    // get_scale_and_shift(effective_output_scale, scale, shift, 32);
    // multiplier[filter.index()] = scale;
    // shift[filter.index()] = shift;
    QuantizeMultiplier(effective_output_scale, &multiplier[filter.index()],
                       &shift[filter.index()]);
  }
  auto ctx = getContext();
  OpBuilder builder(ctx);
  auto op = getOperation();
  builder.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(input());
  auto filter_stype = Module::getStorageType(filter());
  auto filter_new_type =
      RankedTensorType::get(filter_type.getShape(), filter_stype);
  filter().setType(filter_new_type);
  operands.push_back(filter());

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  int32_t input_zeroPoint = input_qtype.getZeroPoint();
  bool with_bias = true;
  if (input_zeroPoint != 0) {
    // merge input_zeroPoint to bias
    std::shared_ptr<std::vector<int32_t>> bias_quant;
    std::shared_ptr<std::vector<int8_t>> filter_quant;
    filter_quant = cast<top::WeightOp>(filter().getDefiningOp()).read<int8_t>();
    if (attr.has_bias) {
      bias_quant = cast<top::WeightOp>(bias().getDefiningOp()).read<int32_t>();
    } else {
      bias_quant->resize(attr.oc, 0);
    }
    int64_t oc = filter_type.getShape()[0];
    int64_t kernel_size = filter_type.getNumElements() / oc;

    for (size_t oc_ind = 0; oc_ind < oc; ++oc_ind) {
      for (size_t kernel_ind = 0; kernel_ind < kernel_size; ++kernel_ind) {
        bias_quant->data()[oc_ind] -=
            input_zeroPoint *
            filter_quant->at(kernel_ind + oc_ind * kernel_size);
      }
    }
    auto bias_type = RankedTensorType::get({oc}, builder.getI32Type());
    auto new_bias =
        top::WeightOp::create(op, "_merge_bias", *bias_quant, bias_type);
    operands.push_back(new_bias);
  } else if (attr.has_bias) {
    auto bias_stype = Module::getStorageType(bias());
    auto bias_new_type =
        RankedTensorType::get(Module::getShape(bias()), bias_stype);
    bias().setType(bias_new_type);
    operands.push_back(bias());
  } else {
    with_bias = false;
    operands.push_back(bias());
  }
  attrs.push_back(
      builder.getNamedAttr("with_bias", builder.getBoolAttr(with_bias)));
  auto newType =
      RankedTensorType::get(Module::getShape(output()), builder.getI32Type());
  auto new_name = Module::getName(op).str() + "_int32";
  auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  auto convOp = builder.create<tpu::Conv2DOp>(name_loc, newType,
                                              ArrayRef<Value>{operands},
                                              ArrayRef<NamedAttribute>{attrs});
  // do requant
  int quant_size = filter_scales.size();
  if (quant_size == 1) {
    return do_requant(op->getLoc(), convOp.output(), output().getType(), true,
                      multiplier[0], shift[0], 0);
  } else {
    std::vector<int32_t> quant(quant_size * 3, 0);
    for (int i = 0; i < quant_size; ++i) {
      quant[i * 3] = multiplier[i];
      quant[i * 3 + 1] = shift[i];
      quant[i * 3 + 2] = output_qtype.getZeroPoint();
    }
    auto quant_type =
        RankedTensorType::get({1, quant_size, 1, 3}, builder.getI32Type());
    auto quantValue = top::WeightOp::create(op, "quant", quant, quant_type);
    return do_requant(op->getLoc(), convOp.output(), quantValue,
                      output().getType(), true, 0);
  }
}
