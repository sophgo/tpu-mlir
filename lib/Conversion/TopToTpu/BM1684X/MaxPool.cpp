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

void MaxPoolLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::MaxPoolOp op) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (op.getKernelShape().size() == 3) {
    lowering_common_f32<tpu::Pool3DOp>(rewriter, op, 2);
  } else if (op.getKernelShape().size() == 2) {
    lowering_common_f32<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::Pool1DOp>(rewriter, op);
  }
}

void MaxPoolLowering::LoweringINT8(PatternRewriter &rewriter, top::MaxPoolOp op,
                                   bool asymmetric) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (op.getKernelShape().size() == 3) {
    lowering_common_int8<tpu::Pool3DOp>(rewriter, op, asymmetric, 2);
  } else if (op.getKernelShape().size() == 2) {
    lowering_common_int8<tpu::Pool2DOp>(rewriter, op, asymmetric);
  } else {
    lowering_common_int8<tpu::Pool1DOp>(rewriter, op, asymmetric);
  }
}

void MaxPoolLowering::LoweringINT4(PatternRewriter &rewriter, top::MaxPoolOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void MaxPoolLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::MaxPoolOp op) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (op.getKernelShape().size() == 3) {
    lowering_common_bf16<tpu::Pool3DOp>(rewriter, op, 2);
  } else if (op.getKernelShape().size() == 2) {
    lowering_common_bf16<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_bf16<tpu::Pool1DOp>(rewriter, op);
  }
}

void MaxPoolLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::MaxPoolOp op) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (op.getKernelShape().size() == 3) {
    lowering_common_f16<tpu::Pool3DOp>(rewriter, op, 2);
  } else if (op.getKernelShape().size() == 2) {
    lowering_common_f16<tpu::Pool2DOp>(rewriter, op);
  } else {
    lowering_common_f16<tpu::Pool1DOp>(rewriter, op);
  }
}

void MaxPoolLowering::LoweringF8(PatternRewriter &rewriter,
                                 top::MaxPoolOp op) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  bool isE4 = module::getMode() == module::Mode::F8E4M3;
  if (op.getKernelShape().size() == 3) {
    lowering_common_f8<tpu::Pool3DOp>(rewriter, op, isE4, 2);
  } else if (op.getKernelShape().size() == 2) {
    lowering_common_f8<tpu::Pool2DOp>(rewriter, op, isE4);
  } else {
    lowering_common_f8<tpu::Pool1DOp>(rewriter, op, isE4);
  }
}

void MaxPoolLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::MaxPoolOp op) const {
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  auto round_mode = get_round_mode(op.getRoundModeAttr().str());
  auto first_round_mode = get_round_mode(op.getFirstRoundModeAttr().str());
  Operation *newOp;
  if (op.getKernelShape().size() == 3) {
    newOp = lowering_common<tpu::Pool3DOp>(rewriter, op,
                                           op.getOutput().getType(), 2);
  } else if (op.getKernelShape().size() == 2) {
    newOp =
        lowering_common<tpu::Pool2DOp>(rewriter, op, op.getOutput().getType());
  } else {
    newOp =
        lowering_common<tpu::Pool1DOp>(rewriter, op, op.getOutput().getType());
  }
  newOp->setAttr("round_mode",
                 tpu::RoundModeAttr::get(op.getContext(), round_mode));
  newOp->setAttr("first_round_mode",
                 tpu::RoundModeAttr::get(op.getContext(), first_round_mode));
}

} // namespace bm1684x
} // namespace tpu_mlir
