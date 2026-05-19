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
  UNREACHABLE_OP("Not Implemented", op);
}
void PackLowering::LoweringINT4(PatternRewriter &rewriter, top::PackOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void PackLowering::LoweringF32(PatternRewriter &rewriter,
                               top::PackOp op) const {
  LoweringQuantized(rewriter, op);
}

void PackLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::PackOp op) const {
  LoweringF32(rewriter, op);
}

void PackLowering::LoweringF16(PatternRewriter &rewriter,
                               top::PackOp op) const {
  LoweringF32(rewriter, op);
}

void PackLowering::LoweringF8(PatternRewriter &rewriter, top::PackOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void PackLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::PackOp op) const {
  const int nInputs = op->getNumOperands();
  assert(nInputs == op.getValuesCount());
  std::vector<Value> operands;

  std::vector<int64_t> shape(module::getShape(op.getOutput()));
  shape[op.getAxis()] = 1;
  auto stype = module::getStorageType(op.getInputs()[0]);
  auto newType = RankedTensorType::get(shape, stype);
  for (int i = 0; i < nInputs; ++i) {
    auto input_reshape = do_reshape(op.getInputs()[i], newType);
    operands.push_back(input_reshape);
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("axis", op.getAxisAttr()));
  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op, op.getOutput().getType(),
                                             operands, attrs);
}

} // namespace bm1684x
} // namespace tpu_mlir
