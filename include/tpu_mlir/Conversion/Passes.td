//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_CONVERSION_PASSES
#define TPU_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TopToTpu
//===----------------------------------------------------------------------===//

def ConvertTopToTpu : Pass<"convert-top-to-tpu", "ModuleOp"> {
  let summary = "Convert top-level Top Ops to Tpu Ops";
  let constructor = "tpu_mlir::createConvertTopToTpu()";
  let dependentDialects = ["tpu_mlir::top::TopDialect", "tpu_mlir::tpu::TpuDialect"];
  let options = [
    Option<"mode", "mode", "std::string", /*default=*/"",
           "default quantization mode: INT8/BF16/F32">,
    Option<"chip", "chip", "std::string", /*default=*/"",
           "chip: cv183x/cv182x/bm1684/bm1684x">,
    Option<"isAsymmetric", "asymmetric", "bool", /*default=*/"false",
           "true for asymmetric quantization, or false for symmetric">,
  ];
}

#endif // TPU_MLIR_CONVERSION_PASSES
