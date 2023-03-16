//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"

namespace tpu_mlir {
namespace cv18xx {

static void splitPool(PatternRewriter &rewriter, Operation *op,
                      MLIRContext *ctx) {
  auto poolOp = dyn_cast<tpu::Pool2DOp>(op);
  if (poolOp.getPoolMode() == tpu::PoolMode::Max) {
    return;
  }
  Value input_val = poolOp.getOperand();
  Value output_val = poolOp.getResult();
  int64_t on, oc, oh, ow;
  module::getNCHW(output_val, on, oc, oh, ow, false);

  uint64_t lmem_size = 32 * 1024;
  int64_t output_size = module::getNumElements(output_val);
  int64_t n, c, ih, iw;
  module::getNCHW(input_val, n, c, ih, iw, false);
  if ((uint64_t)(ih * iw) < ((lmem_size - output_size) / 2) ||
      !(oh == 1 && ow == 1)) {
    return;
  }
  std::string name = module::getName(output_val).str();
  auto elementType_ = output_val.getType().cast<TensorType>().getElementType();
  std::vector<int> h_slices;
  int h_slice_size = (int)(((lmem_size - output_size) / iw) / 2);
  int total_h = ih;
  while (total_h > 0) {
    if (total_h > h_slice_size) {
      total_h -= h_slice_size;
      h_slices.push_back(h_slice_size);
    } else {
      h_slices.push_back(total_h);
      break;
    }
  }
  rewriter.setInsertionPointAfterValue(input_val);
  int offset = 0;
  std::vector<Value> concat_operands;
  for (auto &slice : h_slices) {
    std::vector<Value> slice_operands;
    slice_operands.emplace_back(input_val);
    slice_operands.emplace_back(module::getNoneOp(op));
    std::vector<NamedAttribute> slice_attrs;
    slice_attrs.emplace_back(rewriter.getNamedAttr(
        "offset", rewriter.getI64ArrayAttr({0, 0, offset, 0})));
    slice_attrs.emplace_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1, 1, 1, 1})));
    offset += slice;
    std::string slice_name = "slice_" + name + std::to_string(offset);
    auto slice_loc = NameLoc::get(rewriter.getStringAttr(slice_name));
    auto slice_type = RankedTensorType::get({n, c, slice, iw}, elementType_);
    auto slice_op = rewriter.create<tpu::SliceOp>(slice_loc, slice_type,
                                                  slice_operands, slice_attrs);
    auto slice_out = slice_op.getResult();

    rewriter.setInsertionPointAfterValue(slice_out);
    std::vector<Value> small_pool_operands;
    small_pool_operands.emplace_back(slice_out);
    std::vector<NamedAttribute> small_pool_attrs;
    small_pool_attrs.emplace_back(rewriter.getNamedAttr(
        "kernel_shape", rewriter.getI64ArrayAttr({slice, iw})));
    small_pool_attrs.emplace_back(
        rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));
    small_pool_attrs.emplace_back(
        rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
    small_pool_attrs.emplace_back(rewriter.getNamedAttr(
        "pool_mode", tpu::PoolModeAttr::get(ctx, tpu::PoolMode::Avg)));
    small_pool_attrs.emplace_back(rewriter.getNamedAttr(
        "multiplier",
        rewriter.getSI32IntegerAttr(poolOp.getMultiplier().value())));
    small_pool_attrs.emplace_back(rewriter.getNamedAttr(
        "rshift", rewriter.getI64IntegerAttr(poolOp.getRshift().value())));
    std::string small_pool_name = "pool_" + name + std::to_string(offset);
    auto small_pool_loc = NameLoc::get(rewriter.getStringAttr(small_pool_name));
    auto small_pool_type = RankedTensorType::get({n, c, 1, 1}, elementType_);
    auto small_pool_op = rewriter.create<tpu::Pool2DOp>(
        small_pool_loc, small_pool_type, small_pool_operands, small_pool_attrs);
    auto small_pool_out = small_pool_op.getResult();
    concat_operands.emplace_back(small_pool_out);
    rewriter.setInsertionPointAfterValue(small_pool_out);
  }

  std::vector<NamedAttribute> concat_attrs;
  concat_attrs.emplace_back(
      rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(2)));
  int h_slices_num = h_slices.size();
  std::vector<int64_t> multilpier_arr(h_slices_num, 1);
  std::vector<int64_t> rshifts_arr(h_slices_num, 0);
  concat_attrs.emplace_back(rewriter.getNamedAttr(
      "multipliers",
      rewriter.getI64ArrayAttr(ArrayRef<int64_t>({multilpier_arr}))));
  concat_attrs.emplace_back(rewriter.getNamedAttr(
      "rshifts", rewriter.getI64ArrayAttr(ArrayRef<int64_t>({rshifts_arr}))));
  std::string concat_name = "concat_" + name;
  auto concat_loc = NameLoc::get(rewriter.getStringAttr(concat_name));
  auto concat_type =
      RankedTensorType::get({n, c, h_slices_num, 1}, elementType_);
  auto concat_op = rewriter.create<tpu::ConcatOp>(
      concat_loc, concat_type, concat_operands, concat_attrs);
  auto concat_out = concat_op.getResult();
  rewriter.setInsertionPointAfterValue(concat_out);

  std::vector<NamedAttribute> final_attrs;
  final_attrs.emplace_back(rewriter.getNamedAttr(
      "kernel_shape", rewriter.getI64ArrayAttr({h_slices_num, 1})));
  final_attrs.emplace_back(
      rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));
  final_attrs.emplace_back(
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
  final_attrs.emplace_back(rewriter.getNamedAttr(
      "pool_mode", tpu::PoolModeAttr::get(ctx, tpu::PoolMode::Avg)));
  final_attrs.emplace_back(
      rewriter.getNamedAttr("multiplier", rewriter.getSI32IntegerAttr(1)));
  final_attrs.emplace_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(0)));
  auto final_type = RankedTensorType::get({n, c, 1, 1}, elementType_);
  rewriter.replaceOpWithNewOp<tpu::Pool2DOp>(
      poolOp, final_type, ValueRange{concat_out}, final_attrs);
}

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
        rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));
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
    auto final_op = rewriter.replaceOpWithNewOp<tpu::Pool2DOp>(
        op, newType, ValueRange{poolOp.getInput()}, attrs);
    auto ctx = getContext();
    splitPool(rewriter, final_op.getOperation(), ctx);

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
