//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-abs"

namespace tpu_mlir {
namespace cv18xx {

void AbsLowering::LoweringINT8(PatternRewriter &rewriter, top::AbsOp absOp,
                               bool asymmetric) const {
  lowering_common_int8<tpu::AbsOp>(rewriter, absOp.getOperation(), asymmetric);
}

void AbsLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::AbsOp absOp) const {
  lowering_common_bf16<tpu::AbsOp>(rewriter, absOp.getOperation());
}

}
}
