//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

namespace tpu_mlir {
namespace cv18xx {

void AvgPoolLowering::LoweringINT8(PatternRewriter &rewriter,
                                   top::AvgPoolOp poolOp,
                                   bool asymmetric) const {
  assert(!asymmetric);
  const size_t kernel_size = poolOp.kernel_shape().size();
  auto kernel = Module::getI64Array(poolOp.kernel_shape());
  int64_t kd = kernel_size == 3 ? kernel->at(0) : 1;
  int64_t kh = kernel_size == 3 ? kernel->at(1) : kernel->at(0);
  int64_t kw =
      kernel_size == 3 ? kernel->at(2) : (kernel_size == 2 ? kernel->at(1) : 1);

  auto op = poolOp.getOperation();
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto in_qtype = Quant::getCalibratedType(poolOp.input());
  auto out_qtype = Quant::getCalibratedType(poolOp.output());
  auto in_thr = in_qtype.getMax();
  auto out_thr = out_qtype.getMax();
  if (kernel_size != 3) {
    double scale = in_thr / (out_thr * kh * kw);
    int64_t multiplier, rshift;

    getRShiftAndMultiplierFromQScale(scale, &multiplier, &rshift);
    attrs.push_back(rewriter.getNamedAttr(
        "multiplier", rewriter.getSI32IntegerAttr((int32_t)multiplier)));
    attrs.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));
  } else {
    llvm_unreachable("Not support 3d avg pool now.");
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
    llvm_unreachable("Not support 3d avg pool now.");
  }
}

void AvgPoolLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::AvgPoolOp poolOp) const {
  llvm_unreachable("Not support now.");
}


} // namespace cv18xx
} // namespace tpu_mlir
