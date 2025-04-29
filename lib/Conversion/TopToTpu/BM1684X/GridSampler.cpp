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

static void LoweringGridSampler(PatternRewriter &rewriter,
                                top::GridSamplerOp op, Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  if (module::isWeight(op.getInput())) {
    auto wOp = op.getInput().getDefiningOp<top::WeightOp>();
    auto stype = module::getStorageType(type);
    if (stype.isF16()) {
      operands.push_back(wOp.clone_f16(op));
    } else if (stype.isBF16()) {
      operands.push_back(wOp.clone_bf16(op));
    } else {
      operands.push_back(op.getInput());
    }
  } else {
    operands.push_back(op.getInput());
  }
  if (module::isWeight(op.getGrid())) {
    auto wOp = op.getGrid().getDefiningOp<top::WeightOp>();
    auto stype = module::getStorageType(type);
    if (stype.isF16()) {
      operands.push_back(wOp.clone_f16(op));
    } else if (stype.isBF16()) {
      operands.push_back(wOp.clone_bf16(op));
    } else {
      operands.push_back(op.getGrid());
    }
  } else {
    operands.push_back(op.getGrid());
  }

  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp); // buffer

  rewriter.replaceOpWithNewOp<tpu::GridSamplerOp>(op, type, operands,
                                                  op->getAttrs());
}

void GridSamplerLowering::LoweringF32(PatternRewriter &rewriter,
                                      top::GridSamplerOp op) const {
  int mode = op.getMode();
  auto dims = module::getShape(op.getInput()).size();
  if (dims > 4 && mode == 1) {
    std::vector<NamedAttribute> attrs;
    std::vector<NamedAttribute> cpu_param;
    attrs.emplace_back(rewriter.getNamedAttr(
        "cpu_op_name", rewriter.getStringAttr("grid_sampler")));

    for (auto &attr : op->getAttrs()) {
      cpu_param.push_back(attr);
    }

    attrs.emplace_back(
        rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(cpu_param)));
    rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, op.getOutput().getType(),
                                                   op->getOperands(), attrs);
  } else {
    LoweringGridSampler(rewriter, op, op.getOutput().getType());
  }
}

void GridSamplerLowering::LoweringINT4(PatternRewriter &rewriter,
                                       top::GridSamplerOp op,
                                       bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void GridSamplerLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::GridSamplerOp op,
                                       bool asymmetric) const {
  LoweringF16(rewriter, op);
}

void GridSamplerLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::GridSamplerOp op) const {
  LoweringF32(rewriter, op);
}

void GridSamplerLowering::LoweringF16(PatternRewriter &rewriter,
                                      top::GridSamplerOp op) const {
  auto mode = op.getMode();
  if (mode == 0) {
    auto new_type = getQuantFloatType<mlir::Float16Type>(op.getOutput());
    LoweringGridSampler(rewriter, op, new_type);
  } else {
    LoweringF32(rewriter, op);
  }
}

void GridSamplerLowering::LoweringF8(PatternRewriter &rewriter,
                                     top::GridSamplerOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void GridSamplerLowering::LoweringQuantized(PatternRewriter &rewriter,
                                            top::GridSamplerOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}
} // namespace bm1684x
} // namespace tpu_mlir
