//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir-c/RegisterEverything.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/MLIRContext.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

void mlirRegisterAllDialects(MlirDialectRegistry registry) {
  static_cast<mlir::DialectRegistry *>(registry.ptr)
      ->insert<mlir::func::FuncDialect, top::TopDialect, tpu::TpuDialect,
               mlir::quant::QuantizationDialect>();
}
