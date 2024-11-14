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
  auto p = op.parseParam();
  auto k = p.kd * p.kh * p.kw;
  op->setAttr("pool_mode",
              tpu::PoolModeAttr::get(op->getContext(), tpu::PoolMode::Max));
  if (k <= 225) {
    if (op.getKernelShape().size() == 3) {
      lowering_common_int8<tpu::Pool3DOp>(rewriter, op, false, 2);
    } else if (op.getKernelShape().size() == 2) {
      lowering_common_int8<tpu::Pool2DOp>(rewriter, op);
    } else {
      lowering_common_int8<tpu::Pool1DOp>(rewriter, op);
    }
  } else {
    if (op.getKernelShape().size() == 3) {
      lowering_common_f32<tpu::Pool3DOp>(rewriter, op);
    } else if (op.getKernelShape().size() == 2) {
      lowering_common_f32<tpu::Pool2DOp>(rewriter, op);
    } else {
      lowering_common_f32<tpu::Pool1DOp>(rewriter, op);
    }
  }
}

} // namespace bm1684
} // namespace tpu_mlir
