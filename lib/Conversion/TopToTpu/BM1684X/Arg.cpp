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

/*
  for chip arch that not support 32bit compute, Arg can only use UINT16 or BF16,
  limit to dtype representation feature, can not use BF16 directly when
  encounter some larger shaping input data.

  search ArgOp's preOp dtype, some dtype irrespectively op will be
  ignored(Reshape, Permute, ...), if preOp is InputOp, return it's origin dtype.
*/
Type getEffectiveElementType(Value value) {
  Value current = value;
  while (auto defOp = current.getDefiningOp()) {
    if (auto inputOp = dyn_cast<top::InputOp>(defOp)) {
      return module::getStorageType(inputOp.getInput());
    }
    if (module::isTypeIndependent(defOp)) {
      if (defOp->getNumOperands() > 0) {
        current = defOp->getOperand(0);
        continue;
      }
    }
    break;
  }
  return module::getStorageType(current);
}

void LoweringArg(PatternRewriter &rewriter, top::ArgOp op, Type type) {
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  std::vector<Type> new_types;
  const auto shape = module::getShape(op.getIndices());
  const auto new_type =
      RankedTensorType::get(shape, rewriter.getI32Type()); // indices : Int32
  new_types.push_back(new_type);

  bool use_int = false;
  if (module::isMARS3() || module::isSGTPUV8()) {
    Type elementType = getEffectiveElementType(op.getInput());
    use_int = elementType.isa<IntegerType>();
  }

  if (!module::isNone(op.getValues())) {
    if (module::isMARS3() || module::isSGTPUV8()) {
      auto shape_value = module::getShape(op.getValues());
      if (use_int) {
        new_types.push_back(
            RankedTensorType::get(shape_value, rewriter.getI16Type()));
      } else {
        new_types.push_back(
            RankedTensorType::get(shape_value, rewriter.getBF16Type()));
      }
    } else {
      new_types.push_back(type);
    }
  } else {
    new_types.push_back(op.getValues().getType());
  }
  auto attrs = op->getAttrs();
  NamedAttrList newAttrs(attrs);
  newAttrs.set("use_int_input", rewriter.getBoolAttr(use_int));

  rewriter.replaceOpWithNewOp<tpu::ArgOp>(op, new_types, operands, newAttrs);
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
