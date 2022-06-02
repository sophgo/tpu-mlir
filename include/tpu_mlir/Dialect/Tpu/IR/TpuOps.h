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

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tpu_mlir/Interfaces/InferenceInterface.h"
#include "tpu_mlir/Interfaces/WeightReorderInterface.h"
#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Interfaces/GlobalGenInterface.h"
#include "tpu_mlir/Support/TensorFile.h"
#include "tpu_mlir/Traits/Traits.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOpsDialect.h.inc"

#include "tpu_mlir/Dialect/Tpu/IR/TpuAttr.h.inc"
#define GET_OP_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h.inc"
