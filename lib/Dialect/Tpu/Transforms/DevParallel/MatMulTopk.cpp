//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/DevParallel/Distribute.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/DevParallel/DistributeUtils.h"

namespace tpu_mlir {
namespace tpu {

// MatMul + Elementwise * n + TopK
template <typename MatMulTy>
LogicalResult
MatMulTopK<MatMulTy>::matchAndRewriteImpl(MatMulTy op,
                                          PatternRewriter &rewriter) const {
  if (!isLargeMatMul(op) || module::isOpInDevParallel(op) || !op->hasOneUse()) {
    return failure();
  }
  auto next_op = *op->user_begin();
  while (next_op->template hasTrait<trait::SupportElementwise>() &&
         next_op->hasOneUse()) {
    next_op = *next_op->user_begin();
  }
  auto topk = dyn_cast<tpu::TopKOp>(next_op);
  if (!topk || topk.getK() != 1 || !topk.getValues().use_empty()) {
    return failure();
  }
  // Bingo !!
  distribute(rewriter, op, next_op, tpu::DevPattern::MatMulTopK);
  return success();
}

template LogicalResult
MatMulTopK<tpu::MatMulOp>::matchAndRewriteImpl(tpu::MatMulOp op,
                                               PatternRewriter &rewriter) const;

template LogicalResult MatMulTopK<tpu::A16MatMulOp>::matchAndRewriteImpl(
    tpu::A16MatMulOp op, PatternRewriter &rewriter) const;

template <typename MatMulTy>
void topKSplit(MatMulTy mm, PatternRewriter &rewriter, tpu::DevBeginOp op,
               int64_t num_devices) {
  if (!mm) {
    return;
  }
  auto next_op = *op->user_begin();
  auto filterOp = mm.getOperand(1).template getDefiningOp<top::WeightOp>();
  auto filterShape = module::getShape(filterOp.getOutput());
  auto outputShape = module::getShape(mm.getOutput());
  auto mm_attrs = mm->getAttrs();
  auto has_bias = !module::isNone(mm.getBias());
  auto num_dims = filterShape.size();

  int a16_mm_w_trans = 0;
  int q_group_size = 0;
  auto a16_mm = dyn_cast<tpu::A16MatMulOp>(mm.getOperation());
  if (a16_mm) {
    a16_mm_w_trans = a16_mm.getWTranspose();
    q_group_size = a16_mm.getQGroupSize();
  }
  // auto weight_bits = a16_mm ? a16_mm.getWeightBits() : 16;

  auto N = filterShape[num_dims - 1 - a16_mm_w_trans];

  Operation *end_op = nullptr;
  std::vector<Value> t_operands;
  for (int i = 0; i < num_devices; i++) {
    auto offset = get_splited_offset(N, num_devices, i, 0, q_group_size);
    auto length = get_splited_size(N, num_devices, i, 0, q_group_size);
    auto suffix = std::to_string(i);

    // next_op = cloneColParallelMatMul(rewriter, next_op, cur_out, num_devices,
    // i, 0);

    auto newFilter =
        module::opSliceAxis(rewriter, mm.getOperand(1),
                            num_dims - 1 - a16_mm_w_trans, offset, length);

    std::vector<Value> operands;
    operands.push_back(mm.getInput());
    operands.push_back(newFilter);
    if (a16_mm) {
      auto scale_op = mm.getOperand(2).template getDefiningOp<top::WeightOp>();
      auto sliced_scale = module::opSliceAxis(rewriter, scale_op,
                                              !a16_mm_w_trans, offset, length);
      operands.push_back(sliced_scale);
      auto zp_op = mm.getOperand(3);
      if (zp_op.getType().template dyn_cast<NoneType>()) {
        operands.push_back(rewriter.create<top::NoneOp>(
            module::getLoc(), rewriter.getNoneType()));
      } else {
        auto zp_weight = zp_op.template getDefiningOp<top::WeightOp>();
        auto sliced_zp = module::opSliceAxis(rewriter, zp_weight,
                                             !a16_mm_w_trans, offset, length);
        operands.push_back(sliced_zp);
      }
    }

    if (has_bias) {
      auto new_bias = module::opSliceAxis(rewriter, mm.getBias(), num_dims - 1,
                                          offset, length);
      operands.push_back(new_bias);
    } else {
      operands.push_back(mm.getBias());
    }
    if (!a16_mm) {
      operands.push_back(module::getNoneOp(mm));
      operands.push_back(module::getNoneOp(mm));
    }
    auto new_loc = module::getLocLike(mm.getOutput(), suffix);
    std::vector<int64_t> new_shape = outputShape;
    new_shape[new_shape.size() - 1] = length;
    auto new_type = module::getTypeLike(mm.getOutput(), new_shape);
    rewriter.setInsertionPointAfter(mm);

    auto new_mm =
        rewriter.create<MatMulTy>(new_loc, new_type, operands, mm_attrs);

    Value cur_output = new_mm.getOutput();
    next_op = *mm->user_begin();
    while (!isa<tpu::TopKOp>(next_op)) {
      auto new_op = cloneOp(rewriter, next_op, new_shape, suffix);
      new_op->setOperand(0, cur_output);
      cur_output = new_op->getResult(0);
      next_op = *next_op->user_begin();
    }
    auto topk = cast<tpu::TopKOp>(next_op);
    auto new_op =
        cloneOp(rewriter, next_op, module::getShape(topk.getIndices()), suffix);
    new_op->setOperand(0, cur_output);
    auto new_topk = cast<tpu::TopKOp>(new_op);
    t_operands.push_back(new_topk.getValues());
    auto indices = new_topk.getIndices();
    if (i == 0) {
      t_operands.push_back(indices);
    } else {
      auto new_name = module::getName(indices);
      auto new_loc = NameLoc::get(
          rewriter.getStringAttr(new_name.str() + suffix + "_add"));
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(offset)));
      rewriter.setInsertionPointAfter(new_op);
      auto new_add = rewriter.create<tpu::AddConstOp>(
          new_loc, indices.getType(), ValueRange{indices}, attrs);
      t_operands.push_back(new_add.getOutput());
    }
    if (end_op == nullptr) {
      end_op = *topk.getIndices().user_begin();
    } else {
      assert(end_op == *topk.getIndices().user_begin());
    }
  }
  end_op->setOperands(t_operands);
  eraseForward(rewriter, mm);
  module::removeUnusedOp();
}

template void topKSplit(tpu::MatMulOp mm, PatternRewriter &rewriter,
                        tpu::DevBeginOp op, int64_t num_devices);

template void topKSplit(tpu::A16MatMulOp mm, PatternRewriter &rewriter,
                        tpu::DevBeginOp op, int64_t num_devices);

} // namespace tpu
} // namespace tpu_mlir
