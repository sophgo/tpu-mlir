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

void ScatterNDLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::ScatterNDOp op) const {
  std::vector<Value> operands;
  operands.push_back(op.getInputData());
  auto indices_op = dyn_cast<top::WeightOp>(op.getIndices().getDefiningOp());
  if (indices_op) {
    // convert fp32 indices into int32
    auto indices_data = indices_op.read<float>();
    std::vector<int32_t> indices_int32_v(
        module::getNumElements(op.getIndices()));
    for (int i = 0; i < module::getNumElements(op.getIndices()); ++i) {
      indices_int32_v[i] = static_cast<int32_t>(indices_data->at(i));
    }
    auto new_type = RankedTensorType::get(module::getShape(op.getIndices()),
                                          rewriter.getI32Type());
    i32_array_t indices_int32 =
        std::make_shared<std::vector<int32_t>>(indices_int32_v);
    auto new_indices_op =
        top::WeightOp::create(op, "indices_int32", *indices_int32, new_type);
    operands.push_back(new_indices_op);
  } else {
    operands.push_back(op.getIndices());
  }

  operands.push_back(op.getUpdates());

  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);

  rewriter.replaceOpWithNewOp<tpu::ScatterNDOp>(op, op.getOutput().getType(),
                                                operands, op->getAttrs());
}

static void RequantizeInt8(PatternRewriter &rewriter, top::ScatterNDOp op,
                           Type type, bool asymmetric) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  auto data = op.getInputData();
  auto updates = op.getUpdates();
  auto output = op.getOutput();
  auto new_data = do_transfer(data, output, asymmetric);
  operands.push_back(new_data);
  operands.push_back(op.getIndices());
  auto new_updates = do_transfer(updates, output, asymmetric);
  operands.push_back(new_updates);

  auto noneOp = module::getNoneOp(op);
  // operands.push_back(noneOp); // indices_coeff
  operands.push_back(noneOp); // buffer

  rewriter.replaceOpWithNewOp<tpu::ScatterNDOp>(op, type, operands,
                                                op->getAttrs());
  return;
}

void ScatterNDLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::ScatterNDOp op,
                                     bool asymmetric) const {
  // LoweringF32(rewriter, op);
  if (module::isWeight(op.getInputData())) {
    LoweringF32(rewriter, op);
    return;
  }
  auto new_type = getQuantFloatType(op.getOutput());
  RequantizeInt8(rewriter, op, new_type, asymmetric);
}

static void LoweringScatterND(PatternRewriter &rewriter, top::ScatterNDOp op,
                              Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  if (module::isWeight(op.getInputData())) {
    auto wOp = op.getInputData().getDefiningOp<top::WeightOp>();
    auto stype = module::getStorageType(type);
    if (stype.isF16()) {
      operands.push_back(wOp.clone_f16(op));
    } else if (stype.isBF16()) {
      operands.push_back(wOp.clone_bf16(op));
    } else {
      operands.push_back(op.getInputData());
    }
  } else {
    operands.push_back(op.getInputData());
  }

  if (module::isWeight(op.getIndices())) {
    auto wOp = op.getIndices().getDefiningOp<top::WeightOp>();
    auto wop_type = wOp.getType().cast<RankedTensorType>();
    auto wop_dtype = wop_type.getElementType();
    if (wop_dtype.isF32()) {
      operands.push_back(wOp.clone_int(op));
    } else if (wop_dtype.isInteger(32)) {
      operands.push_back(op.getIndices());
    } else {
      // convert indices into int32
      auto indices_data = wOp.read<float>();
      std::vector<int32_t> indices_int32_v(
          module::getNumElements(op.getIndices()));
      for (int i = 0; i < module::getNumElements(op.getIndices()); ++i) {
        indices_int32_v[i] = static_cast<int32_t>(indices_data->at(i));
      }
      auto new_type = RankedTensorType::get(module::getShape(op.getIndices()),
                                            rewriter.getI32Type());
      i32_array_t indices_int32 =
          std::make_shared<std::vector<int32_t>>(indices_int32_v);
      auto new_indices_op =
          top::WeightOp::create(op, "indices_int32", *indices_int32, new_type);
      operands.push_back(new_indices_op);
    }

  } else {
    operands.push_back(op.getIndices());
  }

  if (module::isWeight(op.getUpdates())) {
    auto wOp = op.getUpdates().getDefiningOp<top::WeightOp>();
    auto stype = module::getStorageType(type);
    if (stype.isF16()) {
      operands.push_back(wOp.clone_f16(op));
    } else if (stype.isBF16()) {
      operands.push_back(wOp.clone_bf16(op));
    } else {
      operands.push_back(op.getUpdates());
    }
  } else {
    operands.push_back(op.getUpdates());
  }

  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp); // buffer

  rewriter.replaceOpWithNewOp<tpu::ScatterNDOp>(op, type, operands,
                                                op->getAttrs());
  return;
}

void ScatterNDLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::ScatterNDOp op,
                                     bool asymmetric) const {
  // auto new_type = getQuantFloatType(op.getOutput());
  // LoweringScatterND(rewriter, op, new_type);
  LoweringF32(rewriter, op);
}

void ScatterNDLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::ScatterNDOp op) const {
  // LoweringF32(rewriter, op);
  auto new_type = getQuantFloatType<mlir::BFloat16Type>(op.getOutput());
  LoweringScatterND(rewriter, op, new_type);
}

void ScatterNDLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::ScatterNDOp op) const {
  // LoweringF32(rewriter, op);
  auto new_type = getQuantFloatType<mlir::Float16Type>(op.getOutput());
  LoweringScatterND(rewriter, op, new_type);
}

void ScatterNDLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::ScatterNDOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ScatterNDLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::ScatterNDOp op) const {
  LoweringF32(rewriter, op);
  // auto new_type = op.getOutput().getType();
  // LoweringScatterND(rewriter, op, new_type);
}

} // namespace bm1684x
} // namespace tpu_mlir
