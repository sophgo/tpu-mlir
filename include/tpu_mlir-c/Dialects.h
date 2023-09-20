//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_C_DIALECTS_H
#define TPU_MLIR_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Top, top);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Tpu, tpu);

#ifdef __cplusplus
}
#endif

#endif // TPU_MLIR_C_DIALECTS_H
