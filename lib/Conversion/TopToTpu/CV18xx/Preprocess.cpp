//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-Preprocess"
namespace tpu_mlir {
namespace cv18xx {

void PreprocessLowering::LoweringINT8(PatternRewriter &rewriter,
                                      top::PreprocessOp op,
                                      bool asymmetric) const {
  lowering_common_int8<tpu::PreprocessOp>(rewriter, op, asymmetric);
}

void PreprocessLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::PreprocessOp op) const {
  lowering_common_bf16<tpu::PreprocessOp>(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
