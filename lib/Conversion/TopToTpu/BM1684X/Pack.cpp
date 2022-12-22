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

void PackLowering::LoweringINT8(PatternRewriter &rewriter, top::PackOp op,
                               bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void PackLowering::LoweringINT4(PatternRewriter &rewriter, top::PackOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void PackLowering::LoweringF32(PatternRewriter &rewriter,
                              top::PackOp op) const {
  llvm_unreachable("Not Implemented");
}

void PackLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::PackOp op) const {
  llvm_unreachable("Not Implemented");
}

void PackLowering::LoweringF16(PatternRewriter &rewriter,
                              top::PackOp op) const {
  llvm_unreachable("Not Implemented");
}

void PackLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::PackOp op) const {
  if (Quant::isUniformQuantized(op.inputs()[0], op.output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  const int nInputs = op->getNumOperands();
  assert(nInputs == op.values_count()); // TODO: nInput==1
  std::vector<Value> operands;

  std::vector<int64_t>shape(Module::getShape(op.output()));
  shape[op.axis()] = 1;
  auto out_stype = Module::getStorageType(op.output());
  auto newType = RankedTensorType::get(shape, out_stype);
  for (int i = 0; i < nInputs; ++i) {
    auto input_reshape = do_reshape(op.inputs()[i], newType);
    operands.push_back(input_reshape);
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("axis", op.axisAttr()));
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.do_reluAttr()));
  attrs.push_back(rewriter.getNamedAttr("relu_limit", op.relu_limitAttr()));

  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op, op.output().getType(), operands,
                                             attrs);
}

} // namespace bm1684x
} // namespace tpu_mlir
