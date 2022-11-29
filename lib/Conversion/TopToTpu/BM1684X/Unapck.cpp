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

void UnpackLowering::LoweringINT8(PatternRewriter &rewriter, top::UnpackOp unpackOp,
                               bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

void UnpackLowering::LoweringF32(PatternRewriter &rewriter,
                              top::UnpackOp unpackOp) const {
  llvm_unreachable("Not Implemented");
}

void UnpackLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::UnpackOp unpackOp) const {
  llvm_unreachable("Not Implemented");
}

void UnpackLowering::LoweringF16(PatternRewriter &rewriter,
                              top::UnpackOp unpackOp) const {
  llvm_unreachable("Not Implemented");
}

void UnpackLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::UnpackOp op) const {
  if (Quant::isUniformQuantized(op.input(), op.outputs()[0]) == false) {
    llvm_unreachable("input output should be quantized");
  }
  std::vector<Value> operands;

  std::vector<int64_t>shape(Module::getShape(op.outputs()[0]));
  shape[op.axis()] = 1;
  auto qtype = Quant::getUniformQuantizedType(op.input());
  auto newType = RankedTensorType::get(shape, qtype);
  std::vector<Type> newTypes(op.num(), newType);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("axis", op.axisAttr()));
  attrs.push_back(rewriter.getNamedAttr("num", op.numAttr()));
  rewriter.setInsertionPointAfter(op);
  auto new_name = Module::getName(op).str() + "_split";
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  auto splitOp = rewriter.create<tpu::SplitOp>(name_loc, newTypes, ValueRange{op.input()}, attrs);

  std::vector<int64_t>oshape(Module::getShape(op.outputs()[0]));
  auto outType = RankedTensorType::get(oshape, qtype);
  for (int i = 0; i < op.num(); ++i) {
    auto out = splitOp.getResult(i);
    rewriter.setInsertionPointAfterValue(out);
    std::vector<NamedAttribute> attrs = {};
    auto newOp = rewriter.create<tpu::ReshapeOp>(op.getLoc(), outType, ValueRange{out}, attrs);
    operands.push_back(newOp.output());
  }
  rewriter.replaceOp(op, operands);
}

} // namespace bm1684x
} // namespace tpu_mlir
