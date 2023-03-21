//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"

namespace tpu_mlir {
namespace tpu {

std::unique_ptr<LgPass> CreateGroupDataMoveOverlapPass();

} // namespace tpu
} // namespace tpu_mlir
