//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-argmax"

namespace tpu_mlir {
namespace cv18xx {

static void LoweringArg(PatternRewriter &rewriter, top::ArgOp &op,
                        bool quant_i8) {
  if (op.getMode() != "ArgMax") {
    llvm_unreachable("Only support ArgMax!");
  }
  for (uint32_t i = 0; i < op.getResults().size(); ++i) {
    if (module::isNone(op.getResult(i))) {
      continue;
    }
    auto next_op = module::getNextOp(op, i);
    if (next_op && !isa<ReturnOp>(next_op)) {
      llvm_unreachable("ArgMax must be last op!");
    }
  }
  bool with_conf = false;
  if (!module::isNone(op.getValues())) {
    with_conf = true;
  }
  auto shape = module::getShape(op.getInput()).vec();
  auto dim = op.getAxis();
  assert(dim == shape.size() - 1);
  int dim_shape = shape[dim];
  dim_shape = (dim_shape + 256 - 1) / 256;
  shape[dim] = dim_shape;
  Type eltType;
  if (quant_i8) {
    eltType = getQuantInt8Type(op.getInput());
  } else {
    eltType = getQuantBF16Type(op.getInput());
  }
  auto result_type = RankedTensorType::get(
      shape, eltType.cast<RankedTensorType>().getElementType());

  std::vector<NamedAttribute> param;
  // create tpu argmax
  std::vector<NamedAttribute> attrs;
  std::vector<Type> result_types;
  attrs.push_back(rewriter.getNamedAttr("axis", op.getAxisAttr()));
  auto name_loc = NameLoc::get(rewriter.getStringAttr(
      module::getName(op.getIndices()).str() + "_trans1"));
  result_types.emplace_back(result_type);
  result_types.emplace_back(module::getNoneOp(op).getResult().getType());
  auto new_op = rewriter.create<tpu::ArgOp>(
      name_loc, result_types, ValueRange{op.getInput()}, op->getAttrs());

  float scale = 1.0f;
  if (with_conf && quant_i8) {
    scale = module::getThreshold(op.getInput()) / 127;
  }
  param.emplace_back(
      rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)));

  // create cpu op
  param.emplace_back(rewriter.getNamedAttr("axis", op.getAxisAttr()));
  attrs.clear();
  attrs.emplace_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("argmax_v3")));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));

  std::vector<Value> operands;
  operands.emplace_back(op.getInput());
  operands.emplace_back(new_op.getIndices());
  std::vector<Type> cpu_result_types;
  cpu_result_types.emplace_back(RankedTensorType::get(
      module::getShape(op.getIndices()), rewriter.getF32Type()));
  if (with_conf) {
    cpu_result_types.emplace_back(RankedTensorType::get(
        module::getShape(op.getValues()), rewriter.getF32Type()));
    auto cpu_op = rewriter.create<tpu::GenericCpuOp>(
        op.getLoc(), cpu_result_types, operands, attrs);
    rewriter.replaceAllUsesWith(op.getResults(), cpu_op.getOutputs());
  } else {
    auto cpu_op = rewriter.create<tpu::GenericCpuOp>(
        op.getIndices().getLoc(), cpu_result_types, operands, attrs);
    rewriter.replaceAllUsesWith(op.getIndices(), cpu_op.getOutputs()[0]);
  }
  rewriter.eraseOp(op);
  return;
}

void ArgLowering::LoweringINT8(PatternRewriter &rewriter, top::ArgOp op,
                               bool asymmetric) const {
  return LoweringArg(rewriter, op, true);
}

void ArgLowering::LoweringBF16(PatternRewriter &rewriter, top::ArgOp op) const {
  return LoweringArg(rewriter, op, false);
}

} // namespace cv18xx
} // namespace tpu_mlir
