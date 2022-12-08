//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

namespace tpu_mlir {
namespace cv18xx {
void ReduceLowering::LoweringINT8(PatternRewriter &rewriter, top::ReduceOp op,
                                  bool asymmetric) const {
  std::vector<Value> operands;
  std::vector<int64_t> rshift_v(1, 0);
  std::vector<int64_t> multiplier_v(1, 1);
  auto mode = op.mode();
  double in_thr, out_thr;
  in_thr = Quant::getThreshold(op.input());
  out_thr = Quant::getThreshold(op.output());
  double qscale = in_thr / out_thr;
  if (mode == "ReduceL2") {
    LoweringBF16(rewriter, op);
    return;
  }
  if (mode == "ReduceMean") {
    // reduce op
    auto axes_val = Module::getI64Array(op.axes());
    auto input_shape = Module::getShape(op.input());
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
  operands.push_back(op.input());
  if (mode != "ReduceL2") {
    auto none = Module::getNoneOp(op);
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
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::ReduceOp>(op, newType, operands, attrs);
}

void ReduceLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::ReduceOp op) const {
  std::vector<Value> operands;
  auto mode = op.mode();
  operands.push_back(op.input());
  if (mode != "ReduceL2") {
    auto none = Module::getNoneOp(op);
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
