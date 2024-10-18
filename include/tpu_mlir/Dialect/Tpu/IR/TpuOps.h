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
#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Interfaces/InplaceInterface.h"
#include "tpu_mlir/Interfaces/GlobalGenInterface.h"
#include "tpu_mlir/Interfaces/TypeInterface.h"
#include "tpu_mlir/Interfaces/IndexingMapsInterface.h"
#include "tpu_mlir/Support/TensorFile.h"
#include "tpu_mlir/Support/AttrStruct.h"
#include "tpu_mlir/Traits/Traits.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOpsDialect.h.inc"

#include "tpu_mlir/Dialect/Tpu/IR/TpuEnum.h.inc"
#define GET_ATTRDEF_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuAttr.h.inc"
#define GET_OP_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h.inc"

namespace tpu_mlir {
namespace tpu {
const conv_attr_t &getConv2DParam(tpu::Conv2DOp &op);
const deconv_attr_t &getDeconvParam(tpu::DeconvOp &op);
const pool_attr_t &getPool2DParam(tpu::Pool2DOp &op);
const slice_attr_t &getSliceParam(tpu::SliceOp &op);

RunMode getRunMode(mlir::func::FuncOp func);
RunMode getRunMode(Operation *op);
} // namespace tpu
} // namespace tpu_mlir
