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

static void set_arccos_attr(PatternRewriter &rewriter, top::ArccosOp op) {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::ARCCOS));
}

void ArccosLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::ArccosOp op) const {
  set_arccos_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void ArccosLowering::LoweringINT8(PatternRewriter &rewriter, top::ArccosOp op,
                                  bool asymmetric) const {
  // this op not suitable for int8 quant cuz slight deviation in the former op
  // would result in great difference in arccos
  set_arccos_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
