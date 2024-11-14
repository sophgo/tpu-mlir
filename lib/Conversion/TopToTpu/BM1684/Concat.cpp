//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void ConcatLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::ConcatOp op) const {
  lowering_common_f32<tpu::ConcatOp>(rewriter, op);
}

void ConcatLowering::LoweringINT8(PatternRewriter &rewriter, top::ConcatOp op,
                                  bool asymmetric) const {
  // lowering_common_int8<tpu::ConcatOp>(rewriter, op, false);

  // checkout whether weight exist
  for (auto in : op.getInputs()) {
    if (module::isWeight(in)) {
      LoweringF32(rewriter, op);
      return;
    }
  }
  auto op_c = op.getOperation();
  std::vector<Value> operands;
  llvm::SmallDenseMap<Value, Value> valueMap;
  for (int i = 0; i < op.getInputs().size(); ++i) {
    auto in = op.getInputs()[i];
    if (valueMap.contains(in)) {
      operands.push_back(valueMap.at(in));
    } else {
      auto new_in = do_transfer(in, op.getOutput(), asymmetric);
      valueMap[in] = new_in;
      operands.push_back(new_in);
    }
  }
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op_c->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(
      rewriter.getNamedAttr("only_merge", rewriter.getBoolAttr(false)));
  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op_c, newType, operands, attrs);
}

} // namespace bm1684
} // namespace tpu_mlir
