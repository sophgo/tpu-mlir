//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

Value top::AvgPoolOp::lowering_int8_bm1684x(bool asymmetric) {
  const size_t kernel_size = kernel_shape().size();
  auto kernel = Module::getI64Array(kernel_shape());
  int64_t kd = kernel_size == 3 ? kernel->at(0) : 1;
  int64_t kh = kernel_size == 3 ? kernel->at(1) : kernel->at(0);
  int64_t kw =
      kernel_size == 3 ? kernel->at(2) : (kernel_size == 2 ? kernel->at(1) : 1);

  auto op = getOperation();
  auto ctx = getContext();
  OpBuilder builder(ctx);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(input(), in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(output(), out_scale, out_zp, asymmetric);
  if (asymmetric == false && kernel_size != 3) {
    assert(in_zp == 0 && out_zp == 0);
    double scale = in_scale / (out_scale * kh * kw);
    int multiplier, rshift;
    get_scale_and_shift(scale, multiplier, rshift, 8);
    attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getI64IntegerAttr(multiplier)));
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
  } else {
    double scale_factor = in_scale / (kd * kh * kw * out_scale);
    double offset_factor = out_zp - in_scale / out_scale * in_zp;
    attrs.push_back(
        builder.getNamedAttr("scale", builder.getF64FloatAttr(scale_factor)));
    attrs.push_back(
        builder.getNamedAttr("offset", builder.getF64FloatAttr(offset_factor)));
  }
  attrs.push_back(builder.getNamedAttr(
      "pool_mode", tpu::PoolModeAttr::get(getContext(), tpu::PoolMode::Avg)));

  builder.setInsertionPointAfter(op);
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  if (kernel_size == 1) {
    auto newOp = builder.create<tpu::Pool1DOp>(getLoc(), newType,
                                                  ValueRange{input()}, attrs);
    return newOp.output();
  } else if (kernel_size == 2) {
    auto newOp = builder.create<tpu::Pool2DOp>(getLoc(), newType,
                                                  ValueRange{input()}, attrs);
    return newOp.output();

  } else {
    auto newOp = builder.create<tpu::Pool3DOp>(getLoc(), newType,
                                                  ValueRange{input()}, attrs);
    return newOp.output();
  }
}

Value top::AvgPoolOp::lowering_f32_bm1684x() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue = lowering_common_float<tpu::Pool3DOp>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue = lowering_common_float<tpu::Pool2DOp>(getOperation());
  } else {
    newValue = lowering_common_float<tpu::Pool1DOp>(getOperation());
  }
  auto op = newValue.getDefiningOp();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  return newValue;
}

Value top::AvgPoolOp::lowering_bf16_bm1684x() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue =
        lowering_common_float<tpu::Pool3DOp, BFloat16Type>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue =
        lowering_common_float<tpu::Pool2DOp, BFloat16Type>(getOperation());
  } else {
    newValue =
        lowering_common_float<tpu::Pool1DOp, BFloat16Type>(getOperation());
  }
  auto op = newValue.getDefiningOp();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  return newValue;
}

Value top::AvgPoolOp::lowering_f16_bm1684x() {
  Value newValue;
  if (kernel_shape().size() == 3) {
    newValue =
        lowering_common_float<tpu::Pool3DOp, Float16Type>(getOperation());
  } else if (kernel_shape().size() == 2) {
    newValue =
        lowering_common_float<tpu::Pool2DOp, Float16Type>(getOperation());
  } else {
    newValue =
        lowering_common_float<tpu::Pool1DOp, Float16Type>(getOperation());
  }
  auto op = newValue.getDefiningOp();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  return newValue;
}

Value top::AvgPoolOp::lowering_quant_bm1684x() {
#if 0
  Builder builder(getContext());
  auto in0_f32 = do_cast(input(), builder.getF32Type(), false);
  auto op = getOperation();
  op->setOperand(0, in0_f32);
  auto type = output().getType();
  auto v = lowering_common_float<tpu::AvgPool2DOp>(op);
  return do_cast(v, type, true);
#else
  if (false == Quant::isUniformQuantized(input(), output())) {
    llvm_unreachable("input output should be quantized");
  }
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(input(), in_scale, in_zp, true);
  Quant::getScaleAndZeroPoint(output(), out_scale, out_zp, true);
  auto kernel = Module::getI64Array(kernel_shape());
  auto kernel_size = kernel->size();
  auto kernel_sum = std::accumulate(kernel->begin(), kernel->end(), 1,
                                    std::multiplies<int64_t>());
  double scale_factor = in_scale / (kernel_sum * out_scale);
  double offset_factor = out_zp - in_scale / out_scale * in_zp;
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(
      builder.getNamedAttr("scale", builder.getF64FloatAttr(scale_factor)));
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getF64FloatAttr(offset_factor)));

  builder.setInsertionPointAfter(op);
  Operation* newOp;
  if (kernel_size == 1) {
    newOp = builder.create<tpu::Pool1DOp>(getLoc(), output().getType(),
                                                  ValueRange{input()}, attrs);
  } else if (kernel_size == 2) {
    newOp = builder.create<tpu::Pool2DOp>(getLoc(), output().getType(),
                                                  ValueRange{input()}, attrs);
  } else {
    newOp = builder.create<tpu::Pool3DOp>(getLoc(), output().getType(),
                                                  ValueRange{input()}, attrs);
  }
  newOp->setAttr("pool_mode",
                 tpu::PoolModeAttr::get(getContext(), tpu::PoolMode::Avg));
  return newOp->getResult(0);
#endif
}
