//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_TPU_BASE
#define TPU_MLIR_TPU_BASE

include "mlir/IR/OpBase.td"

def Tpu_Dialect : Dialect {
  let name = "tpu";
  let summary = "A tpu dialect for the SOPHGO AI chips";
  let cppNamespace = "::tpu_mlir::tpu";
}

#endif
