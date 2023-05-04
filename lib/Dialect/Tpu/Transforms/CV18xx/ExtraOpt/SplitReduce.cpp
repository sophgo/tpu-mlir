//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/ChipOptimize.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;
using namespace tpu_mlir::backend;

#define DEBUG_TYPE "cv18xx_patterns"
namespace tpu_mlir {
namespace cv18xx {

static bool valid_shape(int axis_dims, cvk_fmt_t fmt) {
  int h, w;
  CV18xx::size_to_hw(axis_dims, h, w);
  auto in_shape = CV18xx::tl_shape_t4(1, (int)CV18xx::NPU_NUM, h, w);
  auto out_shape = CV18xx::tl_shape_t4(1, (int)CV18xx::NPU_NUM, 1, 1);
  auto input_size = CV18xx::lmem_tensor_to_size(in_shape, fmt, 1);
  auto output_size = CV18xx::lmem_tensor_to_size(out_shape, fmt, 1);
  return (uint64_t)(input_size) < ((CV18xx::LMEM_BYTES - 2 * output_size) / 2);
}

LogicalResult
SplitReducePattern::matchAndRewrite(tpu::ReduceOp reduceOp,
                                    PatternRewriter &rewriter) const {
  // for ppyoloe reduce 1x48x160x160 --> 1x48x1x1
  if (reduceOp.getMode() == "ReduceL2") {
    return failure();
  }
  std::vector<int64_t> axes_v;
  std::vector<std::vector<int64_t>> outputs_shape_v, new_axes_v;
  auto axes = module::getI64Array(reduceOp.getAxes());
  axes_v.assign(axes->begin(), axes->end());
  int32_t start_axis = axes_v.at(0);
  int32_t end_axis = axes_v.back() + 1;
  auto input_shape = module::getShape(reduceOp.getInput()).vec();
  auto num_axes = axes_v.size();

  int32_t inner_dims =
      std::accumulate(input_shape.begin() + end_axis, input_shape.end(), 1,
                      std::multiplies<int64_t>());
  if (num_axes == 1 || inner_dims > 1) {
    // TODO
    return failure();
  }
  int32_t axis_dims = std::accumulate(input_shape.begin() + start_axis,
                                      input_shape.begin() + end_axis, 1,
                                      std::multiplies<int64_t>());
  auto fmt = CV18xx::getDataType(reduceOp.getInput());
  if (valid_shape(axis_dims, fmt)) {
    return failure();
  }

  for (int32_t i = num_axes - 1; i > 0; i--) {
    int32_t axis = axes_v.at(i);
    axes_v.pop_back();
    new_axes_v.push_back({axis});
    input_shape[axis] = 1;
    outputs_shape_v.push_back(input_shape);
    end_axis = axes_v.back() + 1;
    axis_dims = std::accumulate(input_shape.begin() + start_axis,
                                input_shape.begin() + end_axis, 1,
                                std::multiplies<int64_t>());
    if (valid_shape(axis_dims, fmt)) {
      new_axes_v.push_back(axes_v);
      outputs_shape_v.push_back(module::getShape(reduceOp.getOutput()).vec());
      break;
    }
  }

  if (!valid_shape(axis_dims, fmt)) {
    // TODO. Reshape the reduce op to valid
    llvm_unreachable("reduce's axis_dims is too large.");
  }

  // creat Op
  rewriter.setInsertionPointAfter(reduceOp);
  auto op_name = module::getName(reduceOp.getOperation()).str();
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  auto eltType = module::getElementType(reduceOp.getOutput());
  auto noneOp = module::getNoneOp(reduceOp);
  operands.push_back(reduceOp.getInput());
  operands.push_back(noneOp);
  operands.push_back(noneOp);
  Value newValue = reduceOp.getOutput();
  for (uint32_t i = 0; i < new_axes_v.size(); i++) {
    auto newType = RankedTensorType::get(outputs_shape_v[i], eltType);
    auto name = op_name + "_" + std::to_string(i);
    if (i == new_axes_v.size() - 1) {
      name = op_name;
    }
    auto loc = NameLoc::get(rewriter.getStringAttr(name));
    auto newOp = rewriter.create<tpu::ReduceOp>(loc, newType, operands,
                                                reduceOp->getAttrs());
    newOp->setAttr("axes", rewriter.getI64ArrayAttr(new_axes_v[i]));
    newValue = newOp.getOutput();
    operands[0] = newValue;
  }
  rewriter.replaceOp(reduceOp, {newValue});
  return success();
}
} // namespace cv18xx
} // namespace tpu_mlir
