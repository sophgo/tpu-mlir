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
    Option<"qtable", "qtable", "std::string", /*default=*/"",
           "a table of Ops that quantized to specific mode">,
    Option<"isAsymmetric", "asymmetric", "bool", /*default=*/"false",
           "true for asymmetric quantization, or false for symmetric">,
    Option<"doWinograd", "doWinograd", "bool", /*default=*/"false",
           "true for trying winograd ,or false for not">,
    Option<"ignore_f16_overflow", "ignore_f16_overflow", "bool", /*default=*/"false",
           "some ops convert from f16 to f32, to avoid f16 overflow">,
    Option<"weightFileName", "weightFileName", "std::string", /*default=*/"",
           "weight file name to save">,
    Option<"quantGroupSize", "q_group_size", "int", /*default=*/"",
           "group size for per-group W4A16 quant">,
    Option<"matmulPerchannel", "matmul_perchannel", "bool", /*default=*/"false",
           "true if matmul uses perchannel quant, or false for pertensor quant">,
  ];
}


//===----------------------------------------------------------------------===//
// TopToTosa
//===----------------------------------------------------------------------===//
def ConvertTopToTosa : Pass<"convert-top-to-tosa", "ModuleOp"> {
  let summary = "Convert top-level Top Ops to Tosa Ops";
  let constructor = "tpu_mlir::createConvertTopToTosa()";
  let dependentDialects = ["tpu_mlir::top::TopDialect", "mlir::tosa::TosaDialect"];
  let options = [
    Option<"includeWeight", "includeWeight", "bool", /*default=*/"false",
           "true for including weight datas in tosa.mlir, or false for not">,
  ];
}

//===----------------------------------------------------------------------===//
// TopToLinalg
//===----------------------------------------------------------------------===//
def ConvertTopToLinalg : Pass<"convert-top-to-linalg", "ModuleOp"> {
  let summary = "Convert top-level Top Ops to Linalg Ops";
  let constructor = "tpu_mlir::createConvertTopToLinalg()";
  let dependentDialects = ["tpu_mlir::top::TopDialect", "mlir::linalg::LinalgDialect"];
  let options = [
    Option<"includeWeight", "includeWeight", "bool", /*default=*/"false",
           "true for including weight datas in linalg.mlir, or false for not">,
  ];
}

#endif // TPU_MLIR_CONVERSION_PASSES
