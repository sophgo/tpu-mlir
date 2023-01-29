//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"

namespace tpu_mlir {
namespace tpu {

std::unique_ptr<LgPass> CreateTimeStepCombinePass();

} // namespace tpu
} // namespace tpu_mlir
