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

void DivTryLowering::Lowering(PatternRewriter &rewriter, top::DivOp op) const {
  auto opds = op->getOperands();
  auto with_shape = std::any_of(opds.begin(), opds.end(), [](Value opd) {
    return opd.getDefiningOp()->hasTrait<trait::ShapeProducer>();
  });
  auto all_shape_wight = std::all_of(opds.begin(), opds.end(), [](Value opd) {
    return opd.getDefiningOp()->hasTrait<trait::ShapeProducer>() ||
           module::isWeight(opd);
  });
  if (with_shape && !all_shape_wight) {
    for (uint32_t idx = 0; idx < op->getNumOperands(); idx++) {
      try_insert_host2device(op, idx);
    }
  }
}

void DivLowering::LoweringF32(PatternRewriter &rewriter, top::DivOp op) const {
  lowering_common_f32<tpu::DivOp>(rewriter, op);
}

void DivLowering::LoweringINT8(PatternRewriter &rewriter, top::DivOp op,
                               bool asymmetric) const {
  if (module::isMARS3() || module::isSGTPUV8()) {
    lowering_common_bf16<tpu::DivOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::DivOp>(rewriter, op);
  }
}
void DivLowering::LoweringINT4(PatternRewriter &rewriter, top::DivOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void DivLowering::LoweringBF16(PatternRewriter &rewriter, top::DivOp op) const {
  if (module::isBM1688() || module::isSG2380() || module::isMARS3() ||
      module::isSGTPUV8()) {
    lowering_common_bf16<tpu::DivOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::DivOp>(rewriter, op);
  }
}

void DivLowering::LoweringF16(PatternRewriter &rewriter, top::DivOp op) const {
  if (module::isBM1688() || module::isSG2380()) {
    lowering_common_f16<tpu::DivOp>(rewriter, op);
  } else if (module::isMARS3() || module::isSGTPUV8()) {
    lowering_common_bf16<tpu::DivOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::DivOp>(rewriter, op);
  }
}

void DivLowering::LoweringF8(PatternRewriter &rewriter, top::DivOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void DivLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::DivOp op) const {
  lowering_common<tpu::DivOp>(rewriter, op.getOperation(),
                              op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
