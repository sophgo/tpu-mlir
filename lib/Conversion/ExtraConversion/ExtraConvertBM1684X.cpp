//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertBM1684X.h"

namespace tpu_mlir {

namespace bm1684x {

void populateDoExtraConversionPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      ConvertMatMulWithRightTranspose
  >(patterns->getContext());
  // clang-format on
}

} // namespace bm1684x
} // namespace tpu_mlir
