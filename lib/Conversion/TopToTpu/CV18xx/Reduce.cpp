//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace cv18xx {

static bool valid_shape(int axis_dims, cvk_fmt_t fmt) {
  assert(axis_dims > 0 && "axis_dims should > 0.");
  int h, w;
  CV18xx::size_to_hw(axis_dims, h, w);
  auto in_shape = CV18xx::tl_shape_t4(1, (int)CV18xx::NPU_NUM, h, w);
  auto out_shape = CV18xx::tl_shape_t4(1, (int)CV18xx::NPU_NUM, 1, 1);
  auto input_size = CV18xx::lmem_tensor_to_size(in_shape, fmt, 1);
  auto output_size = CV18xx::lmem_tensor_to_size(out_shape, fmt, 1);
  return (uint64_t)(input_size) < ((CV18xx::LMEM_BYTES - 2 * output_size) / 2);
}

static bool splitReduce(PatternRewriter &rewriter, top::ReduceOp reduceOp) {
  // for ppyoloe reduce 1x48x160x160 --> 1x48x1x1
  if (reduceOp.getMode() == "ReduceL2") {
    return false;
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
    return false;
  }
  int32_t axis_dims = std::accumulate(input_shape.begin() + start_axis,
                                      input_shape.begin() + end_axis, 1,
                                      std::multiplies<int64_t>());

  auto fmt = CV18xx::getDataType(reduceOp.getInput());
  if (valid_shape(axis_dims, fmt)) {
    return false;
  }
  // prevent accuracy issue
  auto eltType = rewriter.getBF16Type();
  fmt = CVK_FMT_BF16;
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
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  // auto eltType = module::getElementType(reduceOp.getOutput());
  auto noneOp = module::getNoneOp(reduceOp);
  operands.push_back(reduceOp.getInput());
  operands.push_back(noneOp);
  operands.push_back(noneOp);
  Value newValue = reduceOp.getOutput();
  for (uint32_t i = 0; i < new_axes_v.size(); i++) {
    auto newType = RankedTensorType::get(outputs_shape_v[i], eltType);
    Location loc = reduceOp.getLoc();
    if (i != new_axes_v.size() - 1) {
      loc = module::getLocLike(reduceOp.getOutput(), std::to_string(i));
    }
    auto newOp = rewriter.create<tpu::ReduceOp>(loc, newType, operands,
                                                reduceOp->getAttrs());
    newOp->setAttr("axes", rewriter.getI64ArrayAttr(new_axes_v[i]));
    newValue = newOp.getOutput();
    operands[0] = newValue;
  }
  rewriter.replaceOp(reduceOp, {newValue});
  return true;
};

void ReduceLowering::LoweringINT8(PatternRewriter &rewriter, top::ReduceOp op,
                                  bool asymmetric) const {
  if (splitReduce(rewriter, op)) {
    return;
  }
  std::vector<Value> operands;
  std::vector<int64_t> rshift_v(1, 0);
  std::vector<int64_t> multiplier_v(1, 1);
  auto mode = op.getMode();
  double in_thr, out_thr;
  in_thr = module::getThreshold(op.getInput());
  out_thr = module::getThreshold(op.getOutput());
  double qscale = in_thr / out_thr;
  if (mode == "ReduceL2") {
    LoweringBF16(rewriter, op);
    return;
  }
  if (mode == "ReduceMean") {
    // reduce op
    auto axes_val = module::getI64Array(op.getAxes());
    auto input_shape = module::getShape(op.getInput());
    int64_t size = 1;
    for (int32_t i = 0; i < axes_val->size(); i++) {
      auto dim = axes_val->at(i);
      assert(static_cast<unsigned>(dim) < input_shape.size() &&
             "Expect valid axis");
      size *= input_shape[dim];
    }
    qscale /= size;
  }
  int64_t multiplier = 0;
  int64_t shift = 0;
  getRShiftAndMultiplierFromQScale(qscale, &multiplier, &shift);
  rshift_v.at(0) = shift;
  multiplier_v.at(0) = multiplier;
  operands.push_back(op.getInput());
  if (mode != "ReduceL2") {
    auto none = module::getNoneOp(op);
    operands.push_back(none);
    operands.push_back(none);
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64ArrayAttr(rshift_v)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::ReduceOp>(op, newType, operands, attrs);
}

void ReduceLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::ReduceOp op) const {
  if (splitReduce(rewriter, op)) {
    return;
  }
  std::vector<Value> operands;
  auto mode = op.getMode();
  operands.push_back(op.getInput());
  if (mode != "ReduceL2") {
    auto none = module::getNoneOp(op);
    operands.push_back(none);
    operands.push_back(none);
  } else {
    int table_h = 32;
    int table_w = 8;
    int table_hw = table_h * table_w;
    std::vector<float> exp_table(table_hw);
    std::vector<float> mantissa_table(table_hw);
    bf16_gen_exponent_mantissa_table("pow", exp_table.data(),
                                     mantissa_table.data(), 0.5f, 0.f);
    auto shape = std::vector<int64_t>{1, 1, table_h, table_w};
    auto table_type = RankedTensorType::get(shape, rewriter.getF32Type());
    auto vtable =
        top::WeightOp::create(op, "reciprocal_table", exp_table, table_type);
    auto vmantissa = top::WeightOp::create(op, "reciprocal_mantissa_table",
                                           mantissa_table, table_type);
    operands.push_back(
        dyn_cast<top::WeightOp>(vtable.getDefiningOp()).clone_bf16(op));
    operands.push_back(
        dyn_cast<top::WeightOp>(vmantissa.getDefiningOp()).clone_bf16(op));
  }
  auto newType = getQuantFloatType<BFloat16Type>(op->getResult(0));
  rewriter.replaceOpWithNewOp<tpu::ReduceOp>(op, newType, operands,
                                             op->getAttrs());
}
} // namespace cv18xx
} // namespace tpu_mlir
