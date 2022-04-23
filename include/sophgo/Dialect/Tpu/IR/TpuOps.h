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
#include "sophgo/Interfaces/LayerGroupInterface.h"
#include "sophgo/Interfaces/CodegenInterface.h"
#include "sophgo/Support/TensorFile.h"
#include "sophgo/Traits/Traits.h"
#include "sophgo/Dialect/Tpu/IR/TpuOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "sophgo/Dialect/Tpu/IR/TpuOps.h.inc"

