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
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Interfaces/WeightReorderInterface.h"
#include "sophgo/Interfaces/LocalGenInterface.h"
#include "sophgo/Interfaces/GlobalGenInterface.h"
#include "sophgo/Support/TensorFile.h"
#include "sophgo/Traits/Traits.h"
#include "sophgo/Dialect/Tpu/IR/TpuOpsDialect.h.inc"

#include "sophgo/Dialect/Tpu/IR/TpuAttr.h.inc"
#define GET_OP_CLASSES
#include "sophgo/Dialect/Tpu/IR/TpuOps.h.inc"
