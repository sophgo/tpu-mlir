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

void ConcatLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::ConcatOp concatOp) const {
  lowering_common_float<tpu::ConcatOp>(rewriter, concatOp.getOperation());
}

void ConcatLowering::LoweringINT8(PatternRewriter &rewriter,
                                  top::ConcatOp concatOp,
                                  bool asymmetric) const {
  auto op = concatOp.getOperation();
  std::vector<Value> operands;
  for (auto in : concatOp.inputs()) {
    auto new_in = do_transfer(in, concatOp.output(), asymmetric);
    operands.push_back(new_in);
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = Quant::getQuantInt8Type(concatOp.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op, newType, operands, attrs);
}

void ConcatLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::ConcatOp concatOp) const {
  lowering_common_float<tpu::ConcatOp, BFloat16Type>(rewriter,
                                                     concatOp.getOperation());
}

void ConcatLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::ConcatOp concatOp) const {
  lowering_common_float<tpu::ConcatOp, Float16Type>(rewriter,
                                                    concatOp.getOperation());
}

void ConcatLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::ConcatOp concatOp) const {
  auto op = concatOp.getOperation();
  std::vector<Value> operands;
  auto out_stype = Module::getStorageType(concatOp.output());
  if (out_stype.isUnsignedInteger(8)) {
    for (auto in : concatOp.inputs()) {
      auto new_in = do_transfer_fp(in, concatOp.output(), true);
      operands.push_back(new_in);
    }
  } else {
    for (auto in : concatOp.inputs()) {
      operands.push_back(in);
    }
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op, concatOp.output().getType(),
                                             operands, attrs);
}

} // namespace bm1684x
} // namespace tpu_mlir
