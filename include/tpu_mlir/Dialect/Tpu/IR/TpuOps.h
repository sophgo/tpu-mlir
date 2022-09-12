//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tpu_mlir/Interfaces/InferenceInterface.h"
#include "tpu_mlir/Interfaces/WeightReorderInterface.h"
#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Interfaces/GlobalGenInterface.h"
#include "tpu_mlir/Support/TensorFile.h"
#include "tpu_mlir/Traits/Traits.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOpsDialect.h.inc"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStep.h"

#include "tpu_mlir/Dialect/Tpu/IR/TpuEnum.h.inc"
#define GET_ATTRDEF_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuAttr.h.inc"
#define GET_OP_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h.inc"
