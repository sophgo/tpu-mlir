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

void LSTMLowering::LoweringF32(PatternRewriter &rewriter,
                               top::LSTMOp op) const {
  auto ctx = getContext();
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  auto lstm_outshape = Module::getShape(op.output());
  std::vector<int64_t> pytorch_lstm_outshape(4, 0);
  pytorch_lstm_outshape[0] = lstm_outshape[0];
  pytorch_lstm_outshape[1] = lstm_outshape[2];
  pytorch_lstm_outshape[2] = lstm_outshape[1];
  pytorch_lstm_outshape[3] = lstm_outshape[3];
  // auto tensor_type = output().getType().cast<RankedTensorType>();
  // tensor_type.setShape(ArrayRef<int64_t>{pytorch_lstm_outshape});
  auto lstmType = RankedTensorType::get(
      ArrayRef<int64_t>{pytorch_lstm_outshape}, rewriter.getF32Type());
  std::string pytorch_lstm_name =
      Module::getName(op.getOperation()).str() + "_pytorch_lstm";
  auto pytorch_lstm = rewriter.getStringAttr(pytorch_lstm_name);
  auto LSTMOp = rewriter.create<tpu::LSTMOp>(
      NameLoc::get(pytorch_lstm), lstmType, ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});

  attrs.clear();
  operands.clear();
  std::vector<int64_t> order = {0, 2, 1, 3};
  attrs.push_back(
      rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
  operands.push_back(LSTMOp.output());
  auto permuteType = RankedTensorType::get(ArrayRef<int64_t>{lstm_outshape},
                                           rewriter.getF32Type());
  rewriter.replaceOpWithNewOp<tpu::PermuteOp>(op, permuteType, operands, attrs);
}

void LSTMLowering::LoweringINT8(PatternRewriter &rewriter,
                                top::LSTMOp op, bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void LSTMLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::LSTMOp op) const {
  LoweringF32(rewriter, op);
}

void LSTMLowering::LoweringF16(PatternRewriter &rewriter,
                               top::LSTMOp op) const {
  LoweringF32(rewriter, op);
}

void LSTMLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::LSTMOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
