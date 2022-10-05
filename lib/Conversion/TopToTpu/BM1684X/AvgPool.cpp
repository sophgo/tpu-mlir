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

void AvgPoolLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::AvgPoolOp poolOp) const {
  auto op = poolOp.getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  if (poolOp.kernel_shape().size() == 3) {
    lowering_common_float<tpu::Pool3DOp>(rewriter, op);
  } else if (poolOp.kernel_shape().size() == 2) {
    lowering_common_float<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_float<tpu::Pool1DOp>(rewriter, op);
  }
}

void AvgPoolLowering::LoweringINT8(PatternRewriter &rewriter,
                                   top::AvgPoolOp poolOp,
                                   bool asymmetric) const {
  const size_t kernel_size = poolOp.kernel_shape().size();
  auto kernel = Module::getI64Array(poolOp.kernel_shape());
  int64_t kd = kernel_size == 3 ? kernel->at(0) : 1;
  int64_t kh = kernel_size == 3 ? kernel->at(1) : kernel->at(0);
  int64_t kw =
      kernel_size == 3 ? kernel->at(2) : (kernel_size == 2 ? kernel->at(1) : 1);

  auto op = poolOp.getOperation();
  auto ctx = getContext();
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(poolOp.input(), in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(poolOp.output(), out_scale, out_zp, asymmetric);
  if (asymmetric == false && kernel_size != 3) {
    assert(in_zp == 0 && out_zp == 0);
    double scale = in_scale / (out_scale * kh * kw);
    int multiplier, rshift;
    get_scale_and_shift(scale, multiplier, rshift, 8);

    attrs.push_back(rewriter.getNamedAttr(
        "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
    attrs.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));
  } else {
    double scale_factor = in_scale / (kd * kh * kw * out_scale);
    double offset_factor = out_zp - in_scale / out_scale * in_zp;
    attrs.push_back(
        rewriter.getNamedAttr("scale", rewriter.getF64FloatAttr(scale_factor)));
    attrs.push_back(rewriter.getNamedAttr(
        "offset", rewriter.getF64FloatAttr(offset_factor)));
  }
  attrs.push_back(rewriter.getNamedAttr(
      "pool_mode", tpu::PoolModeAttr::get(getContext(), tpu::PoolMode::Avg)));

  auto newType = Quant::getQuantInt8Type(poolOp.output(), asymmetric);
  if (kernel_size == 1) {
    rewriter.replaceOpWithNewOp<tpu::Pool1DOp>(
        op, newType, ValueRange{poolOp.input()}, attrs);

  } else if (kernel_size == 2) {
    rewriter.replaceOpWithNewOp<tpu::Pool2DOp>(
        op, newType, ValueRange{poolOp.input()}, attrs);

  } else {
    rewriter.replaceOpWithNewOp<tpu::Pool3DOp>(
        op, newType, ValueRange{poolOp.input()}, attrs);
  }
}

void AvgPoolLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::AvgPoolOp poolOp) const {
  auto op = poolOp.getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  if (poolOp.kernel_shape().size() == 3) {
    lowering_common_float<tpu::Pool3DOp, BFloat16Type>(rewriter, op);
  } else if (poolOp.kernel_shape().size() == 2) {
    lowering_common_float<tpu::Pool2DOp, BFloat16Type>(rewriter, op);
  } else {
    lowering_common_float<tpu::Pool1DOp, BFloat16Type>(rewriter, op);
  }
}

void AvgPoolLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::AvgPoolOp poolOp) const {
  auto op = poolOp.getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  if (poolOp.kernel_shape().size() == 3) {
    lowering_common_float<tpu::Pool3DOp, Float16Type>(rewriter, op);
  } else if (poolOp.kernel_shape().size() == 2) {
    lowering_common_float<tpu::Pool2DOp, Float16Type>(rewriter, op);
  } else {
    lowering_common_float<tpu::Pool1DOp, Float16Type>(rewriter, op);
  }
}

void AvgPoolLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::AvgPoolOp poolOp) const {
  if (false == Quant::isUniformQuantized(poolOp.input(), poolOp.output())) {
    llvm_unreachable("input output should be quantized");
  }
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(poolOp.input(), in_scale, in_zp, true);
  Quant::getScaleAndZeroPoint(poolOp.output(), out_scale, out_zp, true);
  auto kernel = Module::getI64Array(poolOp.kernel_shape());
  auto kernel_size = kernel->size();
  auto kernel_sum = std::accumulate(kernel->begin(), kernel->end(), 1,
                                    std::multiplies<int64_t>());
  double scale_factor = in_scale / (kernel_sum * out_scale);
  double offset_factor = out_zp - in_scale / out_scale * in_zp;
  auto op = poolOp.getOperation();
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(
      rewriter.getNamedAttr("scale", rewriter.getF64FloatAttr(scale_factor)));
  attrs.push_back(
      rewriter.getNamedAttr("offset", rewriter.getF64FloatAttr(offset_factor)));
  attrs.push_back(rewriter.getNamedAttr(
      "pool_mode", tpu::PoolModeAttr::get(getContext(), tpu::PoolMode::Avg)));
  if (kernel_size == 1) {
    rewriter.replaceOpWithNewOp<tpu::Pool1DOp>(
        op, poolOp.output().getType(), ValueRange{poolOp.input()}, attrs);
  } else if (kernel_size == 2) {
    rewriter.replaceOpWithNewOp<tpu::Pool2DOp>(
        op, poolOp.output().getType(), ValueRange{poolOp.input()}, attrs);
  } else {
    rewriter.replaceOpWithNewOp<tpu::Pool3DOp>(
        op, poolOp.output().getType(), ValueRange{poolOp.input()}, attrs);
  }
}

} // namespace bm1684x
} // namespace tpu_mlir
