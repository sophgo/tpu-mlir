//===- TopTFLiteToTpu.h - Convert TOP TFLite to TPU dialect -----*- C++ -*-===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_CONVERSION_TOPTFLITETOTPU_H
#define TPU_MLIR_CONVERSION_TOPTFLITETOTPU_H

#include "mlir/Pass/Pass.h"

namespace tpu_mlir {

void populateTopToTpuConversionPatterns(mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createLowerTopTFLitePass();

} // namespace tpu_mlir

#endif // TPU_MLIR_CONVERSION_TOPTFLITETOTPU_H
