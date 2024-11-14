//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/CAPI/Registration.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Top, top, tpu_mlir::top::TopDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Tpu, tpu, tpu_mlir::tpu::TpuDialect)
