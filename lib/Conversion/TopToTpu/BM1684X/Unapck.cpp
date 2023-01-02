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

void UnpackLowering::LoweringINT8(PatternRewriter &rewriter,
                                  top::UnpackOp unpackOp,
                                  bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void UnpackLowering::LoweringINT4(PatternRewriter &rewriter, top::UnpackOp op,
                                  bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
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
  if (module::isUniformQuantized(op.getInput(), op.getOutputs()[0]) == false) {
    llvm_unreachable("input output should be quantized");
  }
  std::vector<Value> operands;

  std::vector<int64_t> shape(module::getShape(op.getOutputs()[0]));
  shape[op.getAxis()] = 1;
  auto qtype = module::getUniformQuantizedType(op.getInput());
  auto newType = RankedTensorType::get(shape, qtype);
  std::vector<Type> newTypes(op.getNum(), newType);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("axis", op.getAxisAttr()));
  attrs.push_back(rewriter.getNamedAttr("num", op.getNumAttr()));
  rewriter.setInsertionPointAfter(op);
  auto new_name = module::getName(op).str() + "_split";
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  auto splitOp = rewriter.create<tpu::SplitOp>(
      name_loc, newTypes, ValueRange{op.getInput()}, attrs);

  std::vector<int64_t> oshape(module::getShape(op.getOutputs()[0]));
  auto outType = RankedTensorType::get(oshape, qtype);
  for (int i = 0; i < op.getNum(); ++i) {
    auto out = splitOp.getResult(i);
    rewriter.setInsertionPointAfterValue(out);
    std::vector<NamedAttribute> attrs = {};
    auto newOp = rewriter.create<tpu::ReshapeOp>(op.getLoc(), outType,
                                                 ValueRange{out}, attrs);
    operands.push_back(newOp.getOutput());
  }
  rewriter.replaceOp(op, operands);
}

} // namespace bm1684x
} // namespace tpu_mlir
