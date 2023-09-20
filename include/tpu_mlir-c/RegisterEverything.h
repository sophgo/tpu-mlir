//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPUMLIR_C_REGISTER_EVERYTHING_H
#define TPUMLIR_C_REGISTER_EVERYTHING_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Appends all upstream dialects and extensions to the dialect registry.
MLIR_CAPI_EXPORTED void mlirRegisterAllDialects(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // TPUMLIR_C_REGISTER_EVERYTHING_H
