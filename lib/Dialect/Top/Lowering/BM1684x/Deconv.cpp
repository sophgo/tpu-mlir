//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

Value top::DeconvOp::lowering_int8_bm1684x(bool asymmetric) {
  deconv_attr_t param;
  parseParam(&param);
  auto op = getOperation();
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(input());
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
  bool fsign = (fmin < 0 || param.with_bias == true);
  float fqmax = fsign ? 127 : 255;

  std::shared_ptr<std::vector<int32_t>> bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  auto filter_i8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  auto filter_u8 = std::make_shared<std::vector<uint8_t>>(filter_f32->size());
  if (param.with_bias) {
    auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  } else if (in_zp) {
    bias_int32 = std::make_shared<std::vector<int32_t>>(param.oc, 0);
  }

  int inner_dim = filter_f32->size() / param.oc;
  for (int c = 0; c < param.oc; c++) { // per-channel量化
    float *p_filter = filter_f32->data() + c * inner_dim;
    float w_max = findMaxabs(p_filter, inner_dim);
    double scale_w = std::max(w_max / fqmax, 1e-5f);
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
  if (param.with_bias) {
    auto new_type =
        RankedTensorType::get({1, param.oc, 1, 1}, builder.getI32Type());
    auto new_bias = WeightOp::create(op, "bias_int32", *bias_int32, new_type);
    operands.push_back(new_bias);
  } else {
    operands.push_back(bias()); // none
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  std::string deconv_name = Module::getName(op).str() + "_i32";
  auto name = builder.getStringAttr(deconv_name);
  attrs.push_back(
      builder.getNamedAttr("with_bias", builder.getBoolAttr(param.with_bias)));
  auto deconvType = RankedTensorType::get(
      {param.n, param.oc, param.oh, param.ow}, builder.getI32Type());
  auto deconvOp = builder.create<tpu::DeconvOp>(NameLoc::get(name), deconvType,
                                                operands, attrs);

  auto rqType = Quant::getQuantInt8Type(output(), asymmetric);

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
      RankedTensorType::get({1, param.oc, 1, 3}, builder.getI32Type());
  auto new_quant =
      WeightOp::create(deconvOp, "quant_int32", *quant_int32, new_quant_type);

  attrs.clear();
  operands.clear();
  operands.push_back(deconvOp.output());
  operands.push_back(new_quant);
  builder.setInsertionPointAfter(deconvOp);
  auto rqOp =
      builder.create<tpu::RequantIntAxisOp>(op->getLoc(), rqType, operands, attrs);
  return rqOp.output();
}

Value top::DeconvOp::lowering_f32_bm1684x() {
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

  auto newOp = builder.create<tpu::DeconvOp>(op->getLoc(), output().getType(),
                                             operands, attrs);
  return newOp.output();
}

Value top::DeconvOp::lowering_f16_bm1684x() {
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
  auto newOp =
      builder.create<tpu::DeconvOp>(op->getLoc(), newType, operands, attrs);
  return newOp.output();
}

Value top::DeconvOp::lowering_bf16_bm1684x() {
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
  auto newOp =
      builder.create<tpu::DeconvOp>(op->getLoc(), newType, operands, attrs);
  return newOp.output();
}

Value top::DeconvOp::lowering_quant_bm1684x() {
  llvm_unreachable("Not support now");
}
