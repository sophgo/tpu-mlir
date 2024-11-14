//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Support/CustomLayer.h"

namespace tpu_mlir {
namespace bm1684x {

void CustomLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::CustomOp op) const {
  lowering_common_f32<tpu::CustomOp>(rewriter, op, op.getInputs().size() + 1);
}

void CustomLowering::LoweringINT8(PatternRewriter &rewriter, top::CustomOp op,
                                  bool asymmetric) const {
  lowering_common_int8<tpu::CustomOp>(rewriter, op, asymmetric,
                                      op.getInputs().size() + 1);
}

void CustomLowering::LoweringINT4(PatternRewriter &rewriter, top::CustomOp op,
                                  bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void CustomLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::CustomOp op) const {
  lowering_common_bf16<tpu::CustomOp>(rewriter, op, op.getInputs().size() + 1);
}

void CustomLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::CustomOp op) const {
  lowering_common_f16<tpu::CustomOp>(rewriter, op, op.getInputs().size() + 1);
}

void CustomLowering::LoweringF8(PatternRewriter &rewriter,
                                top::CustomOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void CustomLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::CustomOp op) const {
  // lowering_common<tpu::CustomOp>(rewriter, op.getOperation(),
  // op.getOutputs()[0].getType(), op.getInputs().size() + 1);
  auto out_stype = module::getStorageType(op.getOutputs()[0]);
  if (out_stype.isSignedInteger(8)) {
    lowering_common_int8<tpu::CustomOp>(rewriter, op, false,
                                        op.getInputs().size() + 1);
  } else {
    LoweringF16(rewriter, op);
  }
}

} // namespace bm1684x
} // namespace tpu_mlir
