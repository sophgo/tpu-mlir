//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

typedef struct stridedslice_param_t {
  llvm::ArrayRef<int64_t> shape;
  size_t dims;
  uint64_t shrink_axis_mask;
  uint64_t new_axis_mask;
  uint64_t ellipsis_mask;
  uint64_t begin_mask;
  uint64_t end_mask;
  i32_array_t start_v;
  i32_array_t end_v;
  i32_array_t stride_v;
} stridedslice_param_t_t;

static bool parseParam(StridedSliceOp op, stridedslice_param_t_t &param) {
  auto start_op =
      dyn_cast_or_null<top::WeightOp>(op.getStarts().getDefiningOp());
  auto end_op = dyn_cast_or_null<top::WeightOp>(op.getEnds().getDefiningOp());
  auto stride_op =
      dyn_cast_or_null<top::WeightOp>(op.getStrides().getDefiningOp());
  if (!start_op || !end_op || !stride_op) {
    return false;
  }
  param.shape = module::getShape(op.getInput());
  param.dims = param.shape.size();
  param.shrink_axis_mask = op.getShrinkAxisMask();
  param.new_axis_mask = op.getNewAxisMask();
  param.ellipsis_mask = op.getEllipsisMask();
  param.begin_mask = op.getBeginMask();
  param.end_mask = op.getEndMask();
  param.start_v = start_op.read<int32_t>();
  param.end_v = end_op.read<int32_t>();
  param.stride_v = stride_op.read<int32_t>();
  return true;
}

static void rewriteParam(stridedslice_param_t_t param, StridedSliceOp op) {
  op.setShrinkAxisMask(param.shrink_axis_mask);
  op.setNewAxisMask(param.new_axis_mask);
  op.setEllipsisMask(param.ellipsis_mask);
  op.setBeginMask(param.begin_mask);
  op.setEndMask(param.end_mask);
  auto starts_ranked_type = RankedTensorType::get(
      module::getShape(op.getStarts()), module::getStorageType(op.getStarts()));
  auto new_starts =
      top::WeightOp::create(op, "mergerd", *param.start_v, starts_ranked_type);
  op->setOperand(1, new_starts);
  auto ends_ranked_type = RankedTensorType::get(
      module::getShape(op.getEnds()), module::getStorageType(op.getEnds()));
  auto new_ends =
      top::WeightOp::create(op, "mergerd", *param.end_v, ends_ranked_type);
  op->setOperand(2, new_ends);
  auto strides_ranked_type =
      RankedTensorType::get(module::getShape(op.getStrides()),
                            module::getStorageType(op.getStrides()));
  auto new_strides = top::WeightOp::create(op, "mergerd", *param.stride_v,
                                           strides_ranked_type);
  op->setOperand(3, new_strides);
  return;
}

static std::set<int> get_slice_dims(stridedslice_param_t_t param) {
  std::set<int> slice_dims;
  if (param.shrink_axis_mask != 0 || param.new_axis_mask != 0 ||
      param.ellipsis_mask != 0)
    return slice_dims;
  for (size_t i = 0; i < param.dims; i++) {
    int32_t begin = param.start_v->at(i);
    int32_t end = param.end_v->at(i);
    int32_t stride = param.stride_v->at(i);
    if ((param.begin_mask >> i) & 1)
      begin = 0;
    if ((param.end_mask >> i) & 1)
      end = param.shape[i];
    if (stride == 1 && begin == 0 && end == param.shape[i])
      continue;
    slice_dims.insert(i);
  }
  return slice_dims;
}

struct StridedSliceMergePattern : public OpRewriterPatternEx<StridedSliceOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  StridedSliceMergePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<StridedSliceOp>(context,
                                            "StridedSliceMergePattern") {}

  LogicalResult matchAndRewriteImpl(StridedSliceOp op,
                                    PatternRewriter &rewriter) const override {
    if (!op->hasOneUse()) {
      return failure();
    }
    auto in_op = op.getInput().getDefiningOp();
    if (!isa<StridedSliceOp>(in_op) || !in_op->hasOneUse()) {
      return failure();
    }
    stridedslice_param_t_t cur_param;
    if (!parseParam(op, cur_param)) {
      return failure();
    }
    std::set<int> cur_slice_dim = get_slice_dims(cur_param);
    auto pre_op = cast<StridedSliceOp>(in_op);
    stridedslice_param_t_t pre_param;
    if (!parseParam(pre_op, pre_param)) {
      return failure();
    }

    std::set<int> pre_slice_dim = get_slice_dims(pre_param);
    if (pre_slice_dim.empty() || cur_slice_dim.empty()) {
      return failure();
    }
    for (auto d : pre_slice_dim) {
      if (cur_slice_dim.count(d)) {
        return failure();
      }
    }

    for (auto d : pre_slice_dim) {
      cur_param.start_v->at(d) = pre_param.start_v->at(d);
      cur_param.end_v->at(d) = pre_param.end_v->at(d);
      cur_param.stride_v->at(d) = pre_param.stride_v->at(d);
      cur_param.begin_mask = (cur_param.begin_mask & (~(1 << d))) |
                             (pre_param.begin_mask & (1 << d));
      cur_param.end_mask =
          (cur_param.end_mask & (~(1 << d))) | (pre_param.end_mask & (1 << d));
    }
    rewriteParam(cur_param, op);
    op->setOperand(0, pre_op.getInput());
    rewriter.eraseOp(pre_op);
    return success();
  }
};

void StridedSliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<StridedSliceMergePattern>(context);
}
