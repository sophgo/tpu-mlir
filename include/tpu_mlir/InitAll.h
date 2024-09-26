//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Dialect.h"

namespace tpu_mlir {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();
void registerToolPasses();

} // namespace tpu_mlir
