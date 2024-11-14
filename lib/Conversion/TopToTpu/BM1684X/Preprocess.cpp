//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

#define DEBUG_TYPE "lowering-Preprocess"
namespace tpu_mlir {
namespace bm1684x {

void PreprocessLowering::LoweringINT8(PatternRewriter &rewriter,
                                      top::PreprocessOp op,
                                      bool asymmetric) const {
  lowering_common_int8<tpu::PreprocessOp>(rewriter, op, asymmetric, 1);
}

void PreprocessLowering::LoweringF32(PatternRewriter &rewriter,
                                     top::PreprocessOp op) const {
  lowering_common_f32<tpu::PreprocessOp>(rewriter, op, 1);
}

void PreprocessLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::PreprocessOp op) const {
  lowering_common_bf16<tpu::PreprocessOp>(rewriter, op, 1);
}

void PreprocessLowering::LoweringINT4(PatternRewriter &rewriter,
                                      top::PreprocessOp op,
                                      bool asymmetric) const {
  lowering_common_int8<tpu::PreprocessOp>(rewriter, op, asymmetric, 1);
}

void PreprocessLowering::LoweringF16(PatternRewriter &rewriter,
                                     top::PreprocessOp op) const {
  lowering_common_f16<tpu::PreprocessOp>(rewriter, op, 1);
}

void PreprocessLowering::LoweringF8(PatternRewriter &rewriter,
                                    top::PreprocessOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void PreprocessLowering::LoweringQuantized(PatternRewriter &rewriter,
                                           top::PreprocessOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
