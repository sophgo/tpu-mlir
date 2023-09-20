//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

PYBIND11_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "TPU-MLIR Dialects Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
  });
}
