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
  const size_t kernel_size = poolOp.getKernelShape().size();
  auto kernel = module::getI64Array(poolOp.getKernelShape());
  int64_t kh = kernel_size == 3 ? kernel->at(1) : kernel->at(0);
  int64_t kw =
      kernel_size == 3 ? kernel->at(2) : (kernel_size == 2 ? kernel->at(1) : 1);
  if (kernel_size == 3) {
    LoweringBF16(rewriter, poolOp);
    return;
  }
  auto op = poolOp.getOperation();
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto in_qtype = module::getCalibratedType(poolOp.getInput());
  auto out_qtype = module::getCalibratedType(poolOp.getOutput());
  auto in_thr = in_qtype.getMax();
  auto out_thr = out_qtype.getMax();
  if (kernel_size != 3) {
    double scale = in_thr / (out_thr * kh * kw);
    int64_t multiplier, rshift;

    getRShiftAndMultiplierFromQScale(scale, &multiplier, &rshift);
    attrs.push_back(rewriter.getNamedAttr(
        "multiplier", rewriter.getSI32IntegerAttr((int32_t)multiplier)));
    attrs.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshift)));
  } else {
    llvm_unreachable("Not support 3d avg pool now.");
  }
  attrs.push_back(rewriter.getNamedAttr(
      "pool_mode", tpu::PoolModeAttr::get(getContext(), tpu::PoolMode::Avg)));

  auto newType = getQuantInt8Type(poolOp.getOutput(), asymmetric);
  if (kernel_size == 1) {
    rewriter.replaceOpWithNewOp<tpu::Pool1DOp>(
        op, newType, ValueRange{poolOp.getInput()}, attrs);

  } else if (kernel_size == 2) {
    [[maybe_unused]] auto final_op = rewriter.replaceOpWithNewOp<tpu::Pool2DOp>(
        op, newType, ValueRange{poolOp.getInput()}, attrs);
  } else {
    llvm_unreachable("Not support 3d avg pool now.");
  }
}

void AvgPoolLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::AvgPoolOp poolOp) const {
  auto op = poolOp.getOperation();
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Avg));
  if (poolOp.getKernelShape().size() == 3) {
    std::vector<Value> operands;
    auto input_shape = module::getShape(poolOp.getInput());
    auto output_shape = module::getShape(poolOp.getOutput());
    std::vector<int64_t> tmp_shape0(4, 1);
    std::vector<int64_t> tmp_shape1;
    std::vector<int64_t> _kernel;
    std::vector<int64_t> _strides;
    std::vector<int64_t> _pad;

    auto kernel = module::getI64Array(poolOp.getKernelShape());
    auto strides = module::getI64Array(poolOp.getStrides());
    auto pads = module::getI64Array(poolOp.getPads());
    auto type = rewriter.getBF16Type();
    auto op_name = module::getName(poolOp.getOperation()).str();
    // 0. reshape [n c f h w] -> [n*c h w f].
    // It should align_right, this may casuse layerGroup err (fix me)
    module::getNCHW(input_shape, tmp_shape0[0], tmp_shape0[1], tmp_shape0[2],
                    tmp_shape0[3], false);
    auto newType = RankedTensorType::get(tmp_shape0, type);
    auto name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_reshape"));
    auto reshapeOp = rewriter.create<tpu::ReshapeOp>(name_loc, newType,
                                                     poolOp->getOperands());
    // 1. do pool at last 2 dim
    for (int i = 1; i < 3; i++) {
      _kernel.push_back(kernel->at(i));
      _strides.push_back(strides->at(i));
      _pad.push_back(pads->at(i));
    }
    for (int i = 4; i < 6; i++) {
      _pad.push_back(pads->at(i));
    }
    auto dims = input_shape.size();
    tmp_shape0[2] = output_shape[dims - 2];
    tmp_shape0[3] = output_shape[dims - 1];
    newType = RankedTensorType::get(tmp_shape0, type);
    name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_0"));
    auto newOp0 = rewriter.create<tpu::Pool2DOp>(
        name_loc, newType, ValueRange{reshapeOp.getOutput()}, op->getAttrs());
    newOp0->setAttr("kernel_shape", rewriter.getI64ArrayAttr(_kernel));
    newOp0->setAttr("strides", rewriter.getI64ArrayAttr(_strides));
    newOp0->setAttr("pads", rewriter.getI64ArrayAttr(_pad));
    // 2. trans [n*c f h w] -> [n*c h w f]
    std::vector<int64_t> order(tmp_shape0.size());
    std::iota(order.begin(), order.end(), 0);
    order.erase(order.begin() + tmp_shape0.size() - 3);
    order.push_back(tmp_shape0.size() - 3);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
    for (auto i : order) {
      tmp_shape1.push_back(tmp_shape0[i]);
    }
    newType = RankedTensorType::get(tmp_shape1, type);
    name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_trans1"));
    auto newOp1 = rewriter.create<tpu::PermuteOp>(
        name_loc, newType,
        ValueRange{newOp0.getOutput(), module::getNoneOp(op)}, attrs);
    // 3. do pool last dim
    tmp_shape1[tmp_shape1.size() - 1] = output_shape[output_shape.size() - 3];
    newType = RankedTensorType::get(tmp_shape1, type);
    name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_1"));
    auto newOp2 = rewriter.create<tpu::Pool2DOp>(
        name_loc, newType, ValueRange{newOp1.getOutput()}, op->getAttrs());
    newOp2->setAttr("kernel_shape",
                    rewriter.getI64ArrayAttr({1, kernel->at(0)}));
    newOp2->setAttr("strides", rewriter.getI64ArrayAttr({1, strides->at(0)}));
    newOp2->setAttr("pads",
                    rewriter.getI64ArrayAttr({0, pads->at(0), 0, pads->at(3)}));
    // 4. trans back  [n c h w f] -> [n c f h w]
    name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_2"));
    newType = RankedTensorType::get(output_shape, type);
    std::iota(order.begin(), order.end(), 0);
    order.pop_back();
    order.insert(order.begin() + tmp_shape1.size() - 3, tmp_shape1.size() - 1);
    attrs.clear();
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
    auto newOp3 = rewriter.create<tpu::PermuteOp>(
        name_loc, newType,
        ValueRange{newOp2.getOutput(), module::getNoneOp(op)}, attrs);
    // 5. reshape back
    newType = RankedTensorType::get(output_shape, type);
    auto reshape_backOp = rewriter.create<tpu::ReshapeOp>(
        poolOp->getLoc(), newType, ValueRange{newOp3.getOutput()});

    rewriter.replaceOp(op, {reshape_backOp.getOutput()});
  } else if (poolOp.getKernelShape().size() == 2) {
    lowering_common_bf16<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_bf16<tpu::Pool1DOp>(rewriter, op);
  }
}

} // namespace cv18xx
} // namespace tpu_mlir
