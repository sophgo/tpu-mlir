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

static void size_to_hw(int size, int64_t &h, int64_t &w) {
  int div = std::sqrt(size);
  for (h = div; h >= 2; h--) {
    if (size % h == 0) {
      w = size / h;
      break;
    }
  }
}

static bool needReshape(std::vector<int64_t> &shape, std::vector<int64_t> &axes,
                        bool keep_dim) {
  std::vector<int64_t> new_shape(4, 1);
  int num_dims = shape.size();
  int num_axes = axes.size();
  assert(num_axes > 0);
  assert(axes[0] < num_dims && axes[0] >= 0);
  for (int i = 1; i < num_axes; i++) {
    assert(axes[i] == axes[i - 1] + 1);
    assert(axes[i] < num_dims);
  }
  int start_axis = axes[0];
  int end_axis = axes[num_axes - 1] + 1;
  int outer_dims = std::accumulate(shape.begin(), shape.begin() + start_axis, 1,
                                   std::multiplies<int64_t>());
  int axis_dims =
      std::accumulate(shape.begin() + start_axis, shape.begin() + end_axis, 1,
                      std::multiplies<int64_t>());
  int inner_dims = std::accumulate(shape.begin() + end_axis, shape.end(), 1,
                                   std::multiplies<int64_t>());
  if (inner_dims == 1) {
    if (num_dims <= 6 && !keep_dim) {
      return false;
    }
    new_shape[1] = outer_dims;
    size_to_hw(axis_dims, new_shape[2], new_shape[3]);
    axes.clear();
    axes.push_back(2);
    axes.push_back(3);
  } else {
    new_shape[1] = outer_dims;
    new_shape[2] = axis_dims;
    new_shape[3] = inner_dims;
    axes.clear();
    axes.push_back(2);
  }
  shape.clear();
  shape.assign(new_shape.begin(), new_shape.end());
  return true;
}

static void LoweringReduce(PatternRewriter &rewriter, top::ReduceOp op,
                           Type type) {
  auto ctx = rewriter.getContext();
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  Module::getShapeVec(op.input(), input_shape);
  Module::getShapeVec(op.output(), output_shape);
  std::vector<int64_t> axes_val = *Module::getI64Array(op.axes());
  auto op_name = Module::getName(op.getOperation()).str();
  auto output = op->getResult(0);
  auto none = Module::getNoneOp(op);
  Value newValue;
  if (needReshape(input_shape, axes_val, op.keepdims())) {
    // add reshape
    operands.push_back(op.input());
    auto name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_reshape"));
    auto reshapeType = RankedTensorType::get(input_shape, type);
    auto reshapeOp =
        rewriter.create<tpu::ReshapeOp>(name_loc, reshapeType, operands);
    reshapeOp.dump();
    operands.clear();
    operands.push_back(reshapeOp.output());
    operands.push_back(none);
    operands.push_back(none);
    for (int i = axes_val.size() - 1; i >= 0; i--) {
      input_shape.erase(input_shape.begin() + axes_val[i]);
    }
    auto reudceType = RankedTensorType::get(input_shape, type);
    name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_0"));
    auto reduceOp = rewriter.create<tpu::ReduceOp>(name_loc, reudceType,
                                                   operands, op->getAttrs());
    reduceOp->setAttr("axes", rewriter.getI64ArrayAttr(axes_val));
    // reshape back
    operands.clear();
    operands.push_back(reduceOp.output());
    reshapeType = RankedTensorType::get(output_shape, type);
    auto newOp =
        rewriter.create<tpu::ReshapeOp>(op->getLoc(), reshapeType, operands);
    newValue = newOp.output();
  } else {
    Type newType;
    if (type.isF32()) {
      newType = output.getType();
    } else if (type.isF16()) {
      newType = getQuantF16Type(output);
    } else if (type.isBF16()) {
      newType = getQuantBF16Type(output);
    }
    std::vector<Value> operands;
    auto none = Module::getNoneOp(op);
    operands.push_back(op.input());
    operands.push_back(none);
    operands.push_back(none);
    auto newOp = rewriter.create<tpu::ReduceOp>(op->getLoc(), newType, operands,
                                                op->getAttrs());
    newValue = newOp.output();
  }
  rewriter.replaceOp(op, {newValue});
}

void ReduceLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::ReduceOp op) const {
  LoweringReduce(rewriter, op, rewriter.getF32Type());
}

void ReduceLowering::LoweringINT8(PatternRewriter &rewriter, top::ReduceOp op,
                                  bool asymmetric) const {
  LoweringReduce(rewriter, op, rewriter.getF16Type());
}

void ReduceLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::ReduceOp op) const {
  LoweringReduce(rewriter, op, rewriter.getBF16Type());
}

void ReduceLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::ReduceOp op) const {
  LoweringReduce(rewriter, op, rewriter.getF16Type());
}

void ReduceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::ReduceOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
