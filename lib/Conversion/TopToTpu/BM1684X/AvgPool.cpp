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
  if (poolOp.getKernelShape().size() == 3) {
    lowering_common_f32<tpu::Pool3DOp>(rewriter, op, 2);
  } else if (poolOp.getKernelShape().size() == 2) {
    lowering_common_f32<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::Pool1DOp>(rewriter, op);
  }
}

void AvgPoolLowering::LoweringINT4(PatternRewriter &rewriter, top::AvgPoolOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void AvgPoolLowering::LoweringINT8(PatternRewriter &rewriter,
                                   top::AvgPoolOp poolOp,
                                   bool asymmetric) const {
  if (poolOp.getIsAdaptive()) {
    LoweringF32(rewriter, poolOp);
    return;
  }
  auto p = poolOp.parseParam();
  const size_t kernel_size = poolOp.getKernelShape().size();
  int64_t kd = p.kd, kh = p.kh, kw = p.kw;
  auto op = poolOp.getOperation();
  if (asymmetric) {
    // Odd case to f16, [143,143]=>[71,71]
    int ih = (p.oh - 1) * p.sh + p.kh;
    int iw = (p.ow - 1) * p.sw + p.kw;
    if ((iw < p.iw + p.pad_w + p.pad_w_after) ||
        (kernel_size > 1 && ih < p.ih + p.pad_h + p.pad_h_after)) {
      LoweringF16(rewriter, poolOp);
      return;
    }
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  module::getScaleAndZeroPoint(poolOp.getInput(), in_scale, in_zp, asymmetric);
  module::getScaleAndZeroPoint(poolOp.getOutput(), out_scale, out_zp,
                               asymmetric);
  if (asymmetric == false && kernel_size != 3) {
    assert(in_zp == 0 && out_zp == 0);
    double scale = in_scale / (out_scale * kh * kw);
    int multiplier, rshift;
    get_scale_and_shift(scale, multiplier, rshift, 8);

    attrs.push_back(rewriter.getNamedAttr(
        "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
    attrs.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshift)));
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

  auto newType = getQuantInt8Type(poolOp.getOutput(), asymmetric);
  if (kernel_size == 1) {
    rewriter.replaceOpWithNewOp<tpu::Pool1DOp>(
        op, newType, ValueRange{poolOp.getInput()}, attrs);

  } else if (kernel_size == 2) {
    rewriter.replaceOpWithNewOp<tpu::Pool2DOp>(
        op, newType, ValueRange{poolOp.getInput()}, attrs);

  } else {
    auto noneOp = module::getNoneOp(op);
    std::vector<Value> operands;
    int in_num_ops = op->getNumOperands();
    for (int i = 0; i < in_num_ops; ++i) {
      auto in = op->getOperand(i);
      operands.push_back(in);
    }
    operands.push_back(noneOp);
    rewriter.replaceOpWithNewOp<tpu::Pool3DOp>(op, newType, operands, attrs);
  }
}

void AvgPoolLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::AvgPoolOp poolOp) const {
  auto op = poolOp.getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  if (poolOp.getKernelShape().size() == 3) {
    lowering_common_bf16<tpu::Pool3DOp>(rewriter, op, 2);
  } else if (poolOp.getKernelShape().size() == 2) {
    lowering_common_bf16<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_bf16<tpu::Pool1DOp>(rewriter, op);
  }
}

void AvgPoolLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::AvgPoolOp poolOp) const {
  // auto op = poolOp.getOperation();
  // op->setAttr("pool_mode",
  //             tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  // if (poolOp.getKernelShape().size() == 3) {
  //   lowering_common_f16<tpu::Pool3DOp>(rewriter, op, 2);
  // } else if (poolOp.getKernelShape().size() == 2) {
  //   lowering_common_f16<tpu::Pool2DOp>(rewriter, op);
  // } else {
  //   lowering_common_f16<tpu::Pool1DOp>(rewriter, op);
  // }
  LoweringF32(rewriter, poolOp);
}

void AvgPoolLowering::LoweringF8(PatternRewriter &rewriter,
                                 top::AvgPoolOp poolOp) const {
  auto op = poolOp.getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  bool isE4 = module::getMode() == module::Mode::F8E4M3;
  double fp_out_scale;
  assert(op->getNumOperands() == 1);
  double out_scale = module::getCalibratedType(poolOp.getOutput()).getMax();
  double in_scale = module::getCalibratedType(poolOp.getInput()).getMax();
  fp_out_scale = in_scale / out_scale;
  Operation *newOp;
  if (poolOp.getKernelShape().size() == 3) {
    newOp =
        lowering_common_f8<tpu::Pool3DOp>(rewriter, op, isE4, 2).getOperation();
  } else if (poolOp.getKernelShape().size() == 2) {
    newOp =
        lowering_common_f8<tpu::Pool2DOp>(rewriter, op, isE4).getOperation();
  } else {
    newOp =
        lowering_common_f8<tpu::Pool1DOp>(rewriter, op, isE4).getOperation();
  }
  if (isE4)
    newOp->setAttr("fp8_out_scale", rewriter.getF64FloatAttr(fp_out_scale));
}

void AvgPoolLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::AvgPoolOp poolOp) const {
  if (false ==
      module::isUniformQuantized(poolOp.getInput(), poolOp.getOutput())) {
    llvm_unreachable("input output should be quantized");
  }
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  module::getScaleAndZeroPoint(poolOp.getInput(), in_scale, in_zp, true);
  module::getScaleAndZeroPoint(poolOp.getOutput(), out_scale, out_zp, true);
  auto kernel = module::getI64Array(poolOp.getKernelShape());
  auto kernel_size = kernel->size();
  auto kernel_sum = std::accumulate(kernel->begin(), kernel->end(), 1,
                                    std::multiplies<int64_t>());
  double scale_factor = in_scale / (kernel_sum * out_scale);
  double offset_factor = out_zp - in_scale / out_scale * in_zp;
  auto op = poolOp.getOperation();
  auto round_mode = get_round_mode(poolOp.getRoundModeAttr().str());
  auto first_round_mode = get_round_mode(poolOp.getFirstRoundModeAttr().str());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    if (attr.getName() == "round_mode") {
      attrs.push_back(rewriter.getNamedAttr(
          "round_mode",
          tpu::RoundModeAttr::get(poolOp.getContext(), round_mode)));
    } else if (attr.getName() == "first_round_mode") {
      attrs.push_back(rewriter.getNamedAttr(
          "first_round_mode",
          tpu::RoundModeAttr::get(poolOp.getContext(), first_round_mode)));
    } else {
      attrs.push_back(attr);
    }
  }
  attrs.push_back(
      rewriter.getNamedAttr("scale", rewriter.getF64FloatAttr(scale_factor)));
  attrs.push_back(
      rewriter.getNamedAttr("offset", rewriter.getF64FloatAttr(offset_factor)));
  attrs.push_back(rewriter.getNamedAttr(
      "pool_mode", tpu::PoolModeAttr::get(getContext(), tpu::PoolMode::Avg)));
  if (kernel_size == 1) {
    rewriter.replaceOpWithNewOp<tpu::Pool1DOp>(
        op, poolOp.getOutput().getType(), ValueRange{poolOp.getInput()}, attrs);
  } else if (kernel_size == 2) {
    rewriter.replaceOpWithNewOp<tpu::Pool2DOp>(
        op, poolOp.getOutput().getType(), ValueRange{poolOp.getInput()}, attrs);
  } else {
    auto noneOp = module::getNoneOp(op);
    std::vector<Value> operands;
    int in_num_ops = op->getNumOperands();
    for (int i = 0; i < in_num_ops; ++i) {
      auto in = op->getOperand(i);
      operands.push_back(in);
    }
    operands.push_back(noneOp);
    rewriter.replaceOpWithNewOp<tpu::Pool3DOp>(op, poolOp.getOutput().getType(),
                                               operands, attrs);
  }
}

} // namespace bm1684x
} // namespace tpu_mlir
