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

void ConcatTryLowering::Lowering(PatternRewriter &rewriter,
                                 top::ConcatOp op) const {
  const auto opds = op.getOperands();
  bool success = std::any_of(opds.begin(), opds.end(), [](Value opd) {
    return opd.getDefiningOp()->hasTrait<trait::ShapeProducer>();
  });
  if (!success)
    return;
  assert(!op.getDoRelu());
  for (int idx = 0; idx < op.getNumOperands(); ++idx) {
    try_insert_device2host(op.getOperation(), idx);
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(op.getAxis())));
  auto v = op.getResult();
  auto shape = module::getShape(v);
  auto ctx = v.getContext();
  Type new_type = RankedTensorType::get(shape, IntegerType::get(ctx, 32));
  rewriter.replaceOpWithNewOp<tpu::ShapePackOp>(op, new_type, op.getOperands(), attrs);
}

void ConcatLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::ConcatOp op) const {
  lowering_common_f32<tpu::ConcatOp>(rewriter, op);
}
void ConcatLowering::LoweringINT4(PatternRewriter &rewriter, top::ConcatOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ConcatLowering::LoweringINT8(PatternRewriter &rewriter,
                                  top::ConcatOp concatOp,
                                  bool asymmetric) const {
  // checkout whether weight exist
  for (auto in : concatOp.getInputs()) {
    if (module::isWeight(in)) {
      LoweringF16(rewriter, concatOp);
      return;
    }
  }
  auto op = concatOp.getOperation();
  std::vector<Value> operands;
  llvm::SmallDenseMap<Value, Value> valueMap;
  for (auto in : concatOp.getInputs()) {
    if (valueMap.count(in) == 1) {
      operands.push_back(valueMap.lookup(in));
    } else {
      auto new_in = do_transfer(in, concatOp.getOutput(), asymmetric);
      valueMap[in] = new_in;
      operands.push_back(new_in);
    }
  }
  auto newType = getQuantInt8Type(concatOp.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op, newType, operands,
                                             op->getAttrs());
}

void ConcatLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::ConcatOp concatOp) const {
  lowering_common_bf16<tpu::ConcatOp>(rewriter, concatOp.getOperation());
}

void ConcatLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::ConcatOp concatOp) const {
  lowering_common_f16<tpu::ConcatOp>(rewriter, concatOp.getOperation());
}

void ConcatLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::ConcatOp concatOp) const {
  auto op = concatOp.getOperation();
  std::vector<Value> operands;
  auto out_stype = module::getStorageType(concatOp.getOutput());
  if (out_stype.isUnsignedInteger(8)) {
    for (auto in : concatOp.getInputs()) {
      auto new_in = do_transfer_fp(in, concatOp.getOutput(), true);
      operands.push_back(new_in);
    }
  } else {
    for (auto in : concatOp.getInputs()) {
      operands.push_back(in);
    }
  }

  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(op, concatOp.getOutput().getType(),
                                             operands, op->getAttrs());
}

} // namespace bm1684x
} // namespace tpu_mlir
