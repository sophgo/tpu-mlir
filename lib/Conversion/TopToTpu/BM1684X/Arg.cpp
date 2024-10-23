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

void LoweringArg(PatternRewriter &rewriter, top::ArgOp op, Type type) {
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  std::vector<Type> new_types;
  const auto shape = module::getShape(op.getIndices());
  const auto new_type = RankedTensorType::get(shape, rewriter.getI32Type()); //indices : Int32
  new_types.push_back(new_type);
  if (!module::isNone(op.getValues())) {
    if (module::isMARS3()){
      const auto shape_value = module::getShape(op.getValues());
      const auto new_type_value = RankedTensorType::get(shape_value, rewriter.getI16Type()); //value : Int16
      new_types.push_back(new_type_value);
    }else{
      new_types.push_back(type);
    }
  } else {
    new_types.push_back(op.getValues().getType());
  }
  rewriter.replaceOpWithNewOp<tpu::ArgOp>(op, new_types, operands,
                                          op->getAttrs());
  return;
}

void ArgLowering::LoweringF32(PatternRewriter &rewriter, top::ArgOp op) const {
  LoweringArg(rewriter, op, getQuantFloatType(op.getValues()));
}

void ArgLowering::LoweringF16(PatternRewriter &rewriter, top::ArgOp op) const {
  LoweringF32(rewriter, op);
}

void ArgLowering::LoweringBF16(PatternRewriter &rewriter, top::ArgOp op) const {
  LoweringF32(rewriter, op);
}

void ArgLowering::LoweringINT8(PatternRewriter &rewriter, top::ArgOp op,
                               bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void ArgLowering::LoweringINT4(PatternRewriter &rewriter, top::ArgOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void ArgLowering::LoweringF8(PatternRewriter &rewriter, top::ArgOp op) const {
  // llvm_unreachable("FIXME: not implement");
  if (module::getMode() == module::Mode::F8E4M3) {
    LoweringArg(rewriter, op, getQuantF8E4M3Type(op.getValues()));
  } else if (module::getMode() == module::Mode::F8E5M2) {
    LoweringArg(rewriter, op, getQuantF8E5M2Type(op.getValues()));
  }
}

void ArgLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::ArgOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
