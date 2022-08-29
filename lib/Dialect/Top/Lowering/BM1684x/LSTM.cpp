//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

Value top::LSTMOp::lowering_int8_bm1684x(bool asymmetric) {
  llvm_unreachable("LSTMOp unsupported");
  return nullptr;
}

Value top::LSTMOp::lowering_f32_bm1684x() {
  auto ctx = getContext();
  OpBuilder builder(ctx);
  auto op = getOperation();
  builder.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  auto lstm_outshape = Module::getShape(output());
  std::vector<int64_t> pytorch_lstm_outshape(4, 0);
  pytorch_lstm_outshape[0] = lstm_outshape[0];
  pytorch_lstm_outshape[1] = lstm_outshape[2];
  pytorch_lstm_outshape[2] = lstm_outshape[1];
  pytorch_lstm_outshape[3] = lstm_outshape[3];
  //auto tensor_type = output().getType().cast<RankedTensorType>();
  //tensor_type.setShape(ArrayRef<int64_t>{pytorch_lstm_outshape});
  auto lstmType = RankedTensorType::get(ArrayRef<int64_t>{pytorch_lstm_outshape}, builder.getF32Type());
  std::string pytorch_lstm_name = Module::getName(op).str() + "_pytorch_lstm";
  auto pytorch_lstm = builder.getStringAttr(pytorch_lstm_name);
  auto LSTMOp = builder.create<tpu::LSTMOp>(NameLoc::get(pytorch_lstm), lstmType,
                                            ArrayRef<Value>{operands},
                                            ArrayRef<NamedAttribute>{attrs});

  attrs.clear();
  operands.clear();
  std::vector<int64_t> order(4, 0);
  order[0] = 0;
  order[1] = 2;
  order[2] = 1;
  order[3] = 3;
  attrs.push_back(builder.getNamedAttr("order", builder.getI64ArrayAttr(order)));
  operands.push_back(LSTMOp.output());
  builder.setInsertionPointAfter(LSTMOp);
  auto permuteType = RankedTensorType::get(ArrayRef<int64_t>{lstm_outshape}, builder.getF32Type());
  auto permuteOp = builder.create<tpu::PermuteOp>(op->getLoc(), permuteType,
                                                  ArrayRef<Value>{operands},
                                                  ArrayRef<NamedAttribute>{attrs});
  return permuteOp.output();
}

Value top::LSTMOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::LSTMOp, BFloat16Type>(getOperation());
}

Value top::LSTMOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::LSTMOp, Float16Type>(getOperation());
}

Value top::LSTMOp::lowering_quant_bm1684x() {
  llvm_unreachable("LSTMOp unsupported");
  return nullptr;
}
