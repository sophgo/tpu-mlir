//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-abs"

namespace tpu_mlir {
namespace cv18xx {

void AbsLowering::LoweringINT8(PatternRewriter &rewriter, top::AbsOp absOp,
                               bool asymmetric) const {
  auto op = absOp.getOperation();
  op->setAttr("mode", tpu::ActiveModeAttr::get(op->getContext(),
                                               tpu::ActiveMode::ABSVAL));
  lowering_common_int8<tpu::ActiveOp>(rewriter, absOp.getOperation(),
                                      asymmetric);
}

void AbsLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::AbsOp absOp) const {
  auto op = absOp.getOperation();
  op->setAttr("mode", tpu::ActiveModeAttr::get(op->getContext(),
                                               tpu::ActiveMode::ABSVAL));
  lowering_common_bf16<tpu::ActiveOp>(rewriter, absOp.getOperation());
}

} // namespace cv18xx
} // namespace tpu_mlir
