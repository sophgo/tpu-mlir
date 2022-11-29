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
  // const int nInputs = op->getNumOperands();
  // assert(nInputs == op.values_count()); // TODO: nInput==1
  std::vector<Value> operands;

  std::vector<int64_t>shape(Module::getShape(op.input()));
  shape[op.axis()] = 1;
  auto stype = Module::getStorageType(op.input());
  auto newType = RankedTensorType::get(shape, stype);
  std::vector<Type> newTypes(op.num(), newType);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("axis", op.axisAttr()));
  attrs.push_back(rewriter.getNamedAttr("num", op.numAttr()));
  rewriter.setInsertionPointAfter(op);
  auto new_name = Module::getName(op).str() + "_split";
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  auto splitOp = rewriter.create<tpu::SplitOp>(name_loc, newTypes, ValueRange{op.input()}, attrs);

  // rewriter.replaceOpWithNewOp<tpu::SplitOp>(op, op.input().getType(), operands,
  //                                            attrs);

  std::vector<int64_t>oshape(Module::getShape(op.outputs()[0]));
  auto outType = RankedTensorType::get(oshape, stype);
  for (int i = 0; i < op.num(); ++i) {
    auto out = splitOp.getResult(i);
    // auto output_reshape = do_reshape(out, outType);
    rewriter.setInsertionPointAfterValue(out);
    std::vector<NamedAttribute> attrs = {};
    std::string new_name =
        Module::getName(out).str() + "_reshape_" + std::to_string(i);
    auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
    auto newOp = rewriter.create<tpu::ReshapeOp>(name_loc, outType, ValueRange{out}, attrs);
    operands.push_back(newOp.output());
  }
  rewriter.replaceOp(op, operands);
}

} // namespace bm1684x
} // namespace tpu_mlir
